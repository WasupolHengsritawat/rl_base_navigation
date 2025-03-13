#!/usr/bin/python3

import rclpy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, Empty
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import os
import time
import math

GOAL_DISTANCE = 7.5         # m - Distance between start point to goal point
GOAL_POOL_RADIUS = 7        # m - Goal spawn area
NUM_LIDAR_POINTS = 360      # LiDAR points per scan
TIMESTEPS = 3               # Number of timesteps stored
TERMINATION_STEPS = 10000    # Maximum steps 
W_REACH_GOAL = 1            # Weight for reach goal term reward
W_COLLISION = 1             # Weight for collision term reward
W_HIGH_ROTATION = 1         # Weight for high rotation penalty term reward

def random_point_on_arc(h, k, d, r):
    # Compute the condition coefficient
    C = r**2 - h**2 - k**2 - d**2
    if 2 * d == 0:
        return None  # Avoid division by zero
    
    alpha = C / (2 * d)
    
    # Ensure alpha is within valid range
    if alpha < -np.sqrt(h**2 + k**2) or alpha > np.sqrt(h**2 + k**2):
        return None  # No valid points on arc

    # Compute the valid angle range
    theta_min = np.arctan2(k, h) - np.arccos(alpha / np.sqrt(h**2 + k**2))
    theta_max = np.arctan2(k, h) + np.arccos(alpha / np.sqrt(h**2 + k**2))

    # Generate a random theta within the valid range
    theta = np.random.uniform(theta_min, theta_max)

    # Compute the (x, y) coordinates
    x = h + d * np.cos(theta)
    y = k + d * np.sin(theta)

    return np.array([x, y])

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def nth_smallest_unique(arr, n):
    unique_values = np.unique(arr)  # Get unique sorted values
    if n <= 0 or n > len(unique_values):
        raise ValueError("n is out of range")
    return unique_values[n - 1]  # nth smallest (1-based index)

class GazeboEnv(Node, gym.Env):
    def __init__(self):
        rclpy.init()
        Node.__init__(self, 'gazebo_robot_env')

        # Action Space: [linear, angular] per robot
        self.action_space = spaces.Box(low=np.array([0, -0.6]),
                                       high=np.array([0.55, 0.6]), dtype=np.float32)
        
        # Observation Space: LiDAR (360 * 3), Goal (2), Velocity (2) per robot
        self.observation_space = spaces.Box(
            low=np.full((NUM_LIDAR_POINTS * TIMESTEPS + 4), -np.inf),
            high=np.full((NUM_LIDAR_POINTS * TIMESTEPS + 4), np.inf),
            dtype=np.float32
        )

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Change to match publisher
            depth=10
            )
        
        self.cmd_vel_pubs = self.create_publisher(Twist, f'/cmd_vel', 10)
        self.lidar_subs = self.create_subscription(LaserScan, f'/scan', self.lidar_callback, qos_profile) 
        self.odom_subs = self.create_subscription(Odometry, f'/odom', self.odom_callback, 10)
        self.goal_subs = self.create_subscription(Point, f'/goal', self.goal_callback, 10)

        # Data Storage
        self.lidar_data = np.full((TIMESTEPS, NUM_LIDAR_POINTS), 29, dtype=np.float32) # Initialize with maximum range
        self.robot_velocities = np.array([0.0, 0.0])
        self.robot_positions = np.array([0.0, 0.0])
        self.robot_orientations = 0.0
        self.goals = np.array([-8.44, 0.0]) # Default Goal
        self.distance_to_goal = np.array([0.0, 0.0])
        self.distance_to_goal_last = np.array([0.0, 0.0])
        self.num_steps = 0

        # self.start_stage()

    def goal_callback(self, msg):
        self.goals = np.array([msg.x, msg.y])

    def start_episode(self):
        """Start a new Stage simulation."""
        # Initial difference from goal 
        dx = self.goals[0] - self.robot_positions[0]
        dy = self.goals[1] - self.robot_positions[1]
        lin_diff = math.sqrt(dx**2 + dy**2)
        ang_diff = normalize_angle(math.atan2(dy, dx) - self.robot_orientations)

        self.distance_to_goal_last = np.array([lin_diff, ang_diff])
        self.distance_to_goal = self.distance_to_goal_last

    def lidar_callback(self, msg):
        self.lidar_data = np.roll(self.lidar_data, 1, axis=0)
        temp = np.array(msg.ranges, dtype=np.float32)
        temp = np.where(np.isinf(temp), 29, temp)
        self.lidar_data[0] = temp

    def odom_callback(self, msg):
        """Process robot position and velocity data."""
        # Velocities
        self.robot_velocities = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])

        # Positions
        self.robot_positions = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.robot_orientations = math.atan2(siny_cosp, cosy_cosp)

        # Positions difference from goal
        dx = self.goals[0] - self.robot_positions[0]
        dy = self.goals[1] - self.robot_positions[1]
        lin_diff = math.sqrt(dx**2 + dy**2)
        ang_diff = normalize_angle(math.atan2(dy, dx) - self.robot_orientations)

        self.distance_to_goal = np.array([lin_diff, ang_diff])

    def reset(self, seed=None, **kwargs):
        """Reset the environment for a new episode."""
        if seed is not None:
            # Handle seeding logic if necessary (e.g., setting random seed)
            np.random.seed(seed)
            # or any other seed-related logic here
        
        self.start_episode()
        time.sleep(1)  # Allow Stage to initialize
        self.num_steps = 0
        observation = self._get_observation()  # Get the current observation
        info = {}  # You can include any additional info here if needed
        return observation, info

    def step(self, action):
        """Apply action and return observation, reward, done flag."""
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_vel_pubs.publish(twist)

        rclpy.spin_once(self, timeout_sec=0.1)  # Small delay for simulation step

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        truncated = True if self.num_steps > TERMINATION_STEPS else False

        # Debug
        info = {
            "Min Lidar": min(self.lidar_data[0]),
            "Max Lidar": max(self.lidar_data[0]),
            "Mean Lidar": np.average(self.lidar_data[0])
        }

        # self.get_logger().info(f"Num steps: {self.num_steps}")
        self.num_steps += 1

        return obs, reward, done, truncated, info

    def _get_observation(self):
        """Stack last 3 LiDAR scans into a shared observation."""
        obs = []
        for i in range(TIMESTEPS):
            obs.extend(self.lidar_data[i])  # 3 * 360 = 1080
        
        obs.extend(self.distance_to_goal)  # (2,)
        obs.extend(self.robot_velocities)  # (2,)

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        """Compute reward for shared policy training."""
        reward = 0
        
        # Move to the right direction term
        if self.distance_to_goal[0] < 0.2:
            reward += W_REACH_GOAL * 10
        else:
            reward += W_REACH_GOAL * (self.distance_to_goal_last[0] - self.distance_to_goal[0])
        self.distance_to_goal_last = self.distance_to_goal
        
        # Collision term
        min_dist = min(self.lidar_data[0])

        if min_dist == 0:
            min_dist = nth_smallest_unique(self.lidar_data[0],2)
        
        if min_dist < 0.65:  # Collision detected
            reward -= W_COLLISION * 10  # High penalty

        # High rotation penalty term
        if np.abs(self.robot_velocities[1]) > 0.6:
            reward -= W_HIGH_ROTATION * 0.1 * np.abs(self.robot_velocities[1])
        
        return reward

    def _check_done(self):
        """Check if any robot has collided."""

        min_dist = min(self.lidar_data[0])

        if min_dist == 0:
            min_dist = nth_smallest_unique(self.lidar_data[0],2)
    
        if min_dist < 0.65 or self.distance_to_goal[0] < 0.2 or self.num_steps > TERMINATION_STEPS: 
            return True
        
        return False

    def close(self):
        """Shut down ROS and Stage."""
        time.sleep(5)
        os.system("pkill stage_ros")
        os.system("pkill -f dynamic_obstacle_control.py") 
        rclpy.shutdown()


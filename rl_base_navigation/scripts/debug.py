#!/usr/bin/python3

from rl_base_navigation.dummy_module import dummy_function, dummy_var
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan

def nth_smallest_unique(arr, n):
    unique_values = np.unique(arr)  # Get unique sorted values
    if n <= 0 or n > len(unique_values):
        raise ValueError("n is out of range")
    return unique_values[n - 1]  # nth smallest (1-based index)

TIMESTEPS = 3
NUM_LIDAR_POINTS = 360

class DebugNode(Node):
    def __init__(self):
        super().__init__('debug_node')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Change to match publisher
            depth=10
            )

        # self.create_subscription(LaserScan, "/scan", self.laser_callback, qos_profile)
        self.lidar_subs = self.create_subscription(LaserScan, f'/robot_0/base_scan', self.lidar_callback, 10) 

        self.lidar_data = np.full((TIMESTEPS, NUM_LIDAR_POINTS), 29, dtype=np.float32)

    # def laser_callback(self, msg):
    #     self.get_logger().info(f"{len(msg.ranges)}")

    def lidar_callback(self, msg):
        self.lidar_data = np.roll(self.lidar_data, 1, axis=0)
        self.lidar_data[0] = np.array(msg.ranges, dtype=np.float32)

        self.get_logger().info(f"[Lidar] Min: {min(self.lidar_data[0])}, {nth_smallest_unique(self.lidar_data[0],2)}, {nth_smallest_unique(self.lidar_data[0],3)}")
        self.get_logger().info(f"[Lidar] Max {max(self.lidar_data[0])}, Mean {np.average(self.lidar_data[0])}")
        self.get_logger().info(f"------------------------------------------------------------------------------")

def main(args=None):
    rclpy.init(args=args)
    node = DebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
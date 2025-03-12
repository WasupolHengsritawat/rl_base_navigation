#!/usr/bin/python3

import os
import time
import signal
import numpy as np
from ament_index_python.packages import get_package_share_directory
import subprocess

# Constants
NUM_STATIC_OBSTACLES = 7
NUM_DYNAMIC_OBSTACLES = 4
MAX_RADIUS = 7  # Obstacles must be within this range
FORBIDDEN_CIRCLES = np.array([[-5, 0], [5, 0], [0, 5], [0, -5]])  # Forbidden circular areas
FORBIDDEN_RADIUS = 1.2
SAFE_DISTANCE = 0.8  # Minimum distance between obstacles

# Generate a random orientation between -pi and pi
random_orientation = np.random.uniform(-180, 180, 4)

def is_in_forbidden_area(x, y):
    """Check if (x, y) is inside any forbidden zone."""
    for cx, cy in FORBIDDEN_CIRCLES:
        if np.linalg.norm([x - cx, y - cy]) <= FORBIDDEN_RADIUS:
            return True
    return False

def is_too_close(x, y, obstacles, min_distance):
    """Check if (x, y) is too close to existing obstacles."""
    for ox, oy, *_ in obstacles:
        if np.linalg.norm([x - ox, y - oy]) < min_distance:
            return True
    return False

def generate_valid_position(existing_obstacles, min_distance=SAFE_DISTANCE):
    """Generate a valid (x, y) position avoiding forbidden areas and obstacles."""
    while True:
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, MAX_RADIUS)
        x, y = np.round(distance * np.cos(angle), 2), np.round(distance * np.sin(angle), 2)

        if not is_in_forbidden_area(x, y) and not is_too_close(x, y, existing_obstacles, min_distance):
            return x, y

# Generate static obstacles
static_obstacles = []
for _ in range(NUM_STATIC_OBSTACLES):
    x, y = generate_valid_position(static_obstacles)
    theta = np.random.uniform(-np.pi, np.pi)
    size_x = np.round(np.random.uniform(0.2, 1.0), 2)
    size_y = np.round(np.random.uniform(0.2, 1.0), 2)
    static_obstacles.append((x, y, theta, size_x, size_y))

# Generate dynamic obstacles (ensuring they don't collide with static obstacles)
dynamic_obstacles = []
for _ in range(NUM_DYNAMIC_OBSTACLES):
    x, y = generate_valid_position(static_obstacles + dynamic_obstacles)
    theta = np.random.uniform(-np.pi, np.pi)
    size_x = np.round(np.random.uniform(0.2, 1.0), 2)
    size_y = np.round(np.random.uniform(0.2, 1.0), 2)
    speed = np.round(np.random.uniform(0.5, 1.0), 2)
    dynamic_obstacles.append((x, y, theta, size_x, size_y, speed))

# Compute velocity components for dynamic obstacles
dynamic_velocities = [(round(speed * np.cos(theta), 2), round(speed * np.sin(theta), 2)) for _, _, theta, _, _, speed in dynamic_obstacles]

# Define world content
world_content = f"""
include "include/mir_100.inc"

resolution 0.02
interval_sim 100  

define floorplan model
(
  color "gray30"
  boundary 0
  gui_nose 0
  gui_grid 0
  gui_outline 0
  gripper_return 0
  fiducial_return 0
  laser_return 1
)

floorplan
( 
  name "world"
  size [15.000 15.000 0.800]
  pose [0 0 0 0]
  bitmap "bitmaps/circle.jpg"
  gui_move 0
)

mir_100 ( name "robot_0" color "Red" pose [ -5 0 0 {random_orientation[0]} ] )
mir_100 ( name "robot_1" color "Red" pose [  0 5 0 {random_orientation[1]} ] )
mir_100 ( name "robot_2" color "Red" pose [  5 0 0 {random_orientation[2]} ] )
mir_100 ( name "robot_3" color "Red" pose [  0 -5 0 {random_orientation[3]} ] )

define static_obstacle model ( color "gray" )
define dynamic_obstacle position ( color "blue" drive "omni" )
"""

# Add static obstacles
for x, y, theta, size_x, size_y in static_obstacles:
    world_content += f"""
static_obstacle
(
  pose [{x} {y} 0 {theta}]
  size [{size_x} {size_y} 0.8]
)
"""

# Add dynamic obstacles
for i, ((x, y, theta, size_x, size_y, speed), (vel_x, vel_y)) in enumerate(zip(dynamic_obstacles, dynamic_velocities)):
    world_content += f"""
dynamic_obstacle
(
  name "dynamic_obstacle_{i}"
  pose [{x} {y} 0 {theta}]
  size [{size_x} {size_y} 0.8]
)
"""

# Save the world file
stage_world_dir = os.path.join(get_package_share_directory('stage_ros2'), 'world')
stage_world_file_path = os.path.join(stage_world_dir, 'random_env.world')

with open(stage_world_file_path, "w") as world_file:
    world_file.write(world_content)

# Kill previous Stage and controller
os.system("pkill stage_ros")  

# Launch Stage
subprocess.Popen(["ros2", "launch", "stage_ros2", "stage.launch.py", "world:=random_env"])
# os.system("ros2 launch stage_ros2 stage.launch.py world:=simple_env")

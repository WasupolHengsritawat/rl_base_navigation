#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
import numpy as np

class DynamicObstacleController(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_controller')
        
        self.num_obstacles = 4  # Number of dynamic obstacles
        self.pubs = []
        self.speeds = []
        self.directions = []

        # Create service server for rerandom the dynamic obstacle direction
        self.create_service(SetBool, f"/rerandom_req", self.rerandom_callback)

        # Create publishers for each dynamic obstacle
        for i in range(self.num_obstacles):
            pub = self.create_publisher(Twist, f'/dynamic_obstacle_{i}/cmd_vel', 10)
            self.pubs.append(pub)

            # Random speed between 0.5 and 1.0 m/s
            speed = np.random.uniform(0.5, 1.0)
            self.speeds.append(speed)

            # Random direction (angle in radians)
            direction = np.random.uniform(-np.pi, np.pi)
            self.directions.append(direction)

        # Timer to publish commands at 10 Hz
        self.timer = self.create_timer(0.1, self.update_obstacles)

    def update_obstacles(self):
        """Publishes velocity commands to move obstacles in a straight line."""
        for i in range(self.num_obstacles):
            msg = Twist()
            msg.linear.x = self.speeds[i] * np.cos(self.directions[i])  # Velocity X
            msg.linear.y = self.speeds[i] * np.sin(self.directions[i])  # Velocity Y

            self.pubs[i].publish(msg)

            # self.get_logger().info(f'Obstacle {i}: vx={msg.linear.x:.2f}, vy={msg.linear.y:.2f}')
    
    def rerandom_callback(self, req, res):
        if req.data:
            self.speeds = []
            self.directions = []

            for _ in range(self.num_obstacles):
                # Random speed between 0.5 and 1.0 m/s
                speed = np.random.uniform(0.5, 1.0)
                self.speeds.append(speed)

                # Random direction (angle in radians)
                direction = np.random.uniform(-np.pi, np.pi)
                self.directions.append(direction)
            
        response = SetBool.Response()
        response.success = True
        response.message = ""

        return response

def main(args=None):
    rclpy.init()
    node = DynamicObstacleController()
    node.get_logger().info("Dynamic Obstacle Control Node Started")
    rclpy.spin(node)
    node.get_logger().info("Dynamic Obstacle Control Node Exiting")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

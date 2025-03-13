#!/usr/bin/python3

import rclpy
from stable_baselines3 import SAC
from gazebo_gym_env import GazeboEnv

import code
import subprocess
import os
from ament_index_python.packages import get_package_prefix

MODEL_NAME = "stage_random_env_object_avoidance_3"

if __name__ == '__main__':

    # Create a single robot environment
    env = GazeboEnv()

    # Load trained model
    model = SAC.load(MODEL_NAME, env=env)

    # Reset environment
    obs, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False

    step_count = 1
    # Run one episode
    while True:
        print(f"step count: {step_count}")
        step_count += 1

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)  # Unpack properly if needed
        
        total_reward += reward
        print(f"Step Reward: {reward}, Total Reward: {total_reward}, Action: {action}")
        print(info)


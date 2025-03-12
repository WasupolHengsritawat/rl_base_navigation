#!/usr/bin/python3

import rclpy
from stable_baselines3 import SAC
from stage_simple_gym_env import SimpleStageEnv

import code
import subprocess
import os
from ament_index_python.packages import get_package_prefix

MODEL_NAME = "stage_random_env_object_avoidance_1"

pkg_scripts_dir = os.path.join(get_package_prefix('rl_base_navigation'), 'lib', 'rl_base_navigation')
generate_simple_env_file_path = os.path.join(pkg_scripts_dir, 'generate_stage_random_env.py')

if __name__ == '__main__':
    # Start dynamic obstacle control
    subprocess.Popen(["ros2", "run", "rl_base_navigation", "dynamic_obstacle_control.py"])

    os.system(f"python3 {generate_simple_env_file_path}")  # Regenerate world
    subprocess.Popen(["ros2", "launch", "stage_ros2", "stage.launch.py", "world:=random_env"])

    # Create a single robot environment
    env = SimpleStageEnv(robot_id=0)

    # Load trained model
    model = SAC.load(MODEL_NAME, env=env)

    # Reset environment
    obs, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False

    step_count = 1
    # Run one episode
    while not done and not (truncated):
        print(f"step count: {step_count}")
        step_count += 1

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)  # Unpack properly if needed
        
        total_reward += reward
        print(f"Step Reward: {reward}, Total Reward: {total_reward}, Action: {action}")
        print(info)

    print(f"Episode finished with total reward: {total_reward}")

    # # code.interact(local=locals())

    env.close()  # Ensure proper shutdown

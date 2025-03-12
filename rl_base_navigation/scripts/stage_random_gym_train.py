#!/usr/bin/python3

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecEnvWrapper
from stage_random_gym_env import StageEnv 
import subprocess
import torch as th
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor with Conv1D layers for processing LiDAR data."""
    def __init__(self, observation_space, features_dim=184):  # Ensure feature_dim = 132
        super().__init__(observation_space, features_dim)
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(89 * 32, 180)  # Output 128 features from conv layers
    
    def forward(self, observations):
        lidar = observations[:, :360*3].view(-1, 3, 360)  # Reshape LiDAR data
        goal = observations[:, 360*3:360*3+2]  # Goal (2D)
        velocity = observations[:, 360*3+2:360*3+4]  # Velocity (2D)
        
        x = th.relu(self.conv1(lidar))
        x = th.relu(self.conv2(x))
        x = th.flatten(x, start_dim=1)  # Flatten to feed into FC layers
        x = th.relu(self.fc1(x))
        x = th.cat([x, goal, velocity], dim=1)  # Concatenate additional inputs
        return x


class CustomActor(Actor):
    """Custom policy (actor) network using shared feature extractor."""
    def __init__(self, observation_space, action_space, net_arch=None, **kwargs):
        super().__init__(observation_space, action_space, net_arch=net_arch, **kwargs)
        self.features_extractor = CustomFeatureExtractor(observation_space, features_dim=184)


class CustomCritic(ContinuousCritic):
    """Custom critic (value) network with action input concatenation."""
    def __init__(self, observation_space, action_space, net_arch=None, **kwargs):
        super().__init__(observation_space, action_space, net_arch=net_arch, **kwargs)
        self.features_extractor = CustomFeatureExtractor(observation_space, features_dim=184)

    def forward(self, observations, actions):
        x = self.features_extractor(observations)
        x = th.cat([x, actions], dim=1)  # Concatenate action input
        x = self.q_net(x)
        return x
    
class SyncResetVecEnv(VecEnvWrapper):
    """Custom wrapper to synchronize resets across all agents in SubprocVecEnv."""
    def __init__(self, venv):
        super().__init__(venv)
        self.num_envs = venv.num_envs

    def reset(self):
        """Ensure all agents reset at the same time."""
        return self.venv.reset()

    def step_wait(self):
        """Check if any agent is done; if so, reset all agents."""
        obs, rewards, dones, infos = self.venv.step_wait()

        if np.any(dones):  # If any agent is done, reset all agents
            obs = self.reset()

        return obs, rewards, dones, infos

def make_env(robot_id):
    """Creates an environment instance wrapped with Monitor"""
    def _init():
        env = StageEnv(robot_id)
        return Monitor(env)  # Wrap environment with Monitor to log rewards
    return _init


if __name__ == '__main__':
    rclpy.init()
    
    subprocess.Popen(["ros2", "run", "rl_base_navigation", "dynamic_obstacle_control.py"])

    # Create vectorized environment with monitoring
    env = SubprocVecEnv([make_env(i) for i in range(4)])  # Multi-agent environment
    env = SyncResetVecEnv(env)  # Wrap with synchronized reset wrapper
    env = VecMonitor(env)  # Track episode rewards and lengths

    # env = SubprocVecEnv([make_env(i) for i in range(1)])
    # env = VecMonitor(env)  # Track episode rewards and lengths

    # TensorBoard logging setup
    tensorboard_log_dir = "./tensorboard_logs"

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=184),
        net_arch=[128, 128]
    )

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_log_dir
    )

    # Training
    total_timesteps = 50000
    model.learn(total_timesteps=total_timesteps, log_interval=10, progress_bar=True)

    # Save trained model
    model.save("stage_random_env_object_avoidance")

    env.close()
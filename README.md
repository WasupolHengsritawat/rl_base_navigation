# Reinforcement Learning Based Navigation for MiR100

**rl_base_navigation** provides a tool for developing an end-to-end local planner with dynamic object avoidance using Soft Actor-Critic (SAC), specifically designed for the MiR100 robot in a ROS2 environment. The planner receives 360° LiDAR readings (with 360 range data from the /scan topic: LaserScan), the linear and angular distance between the robot and the goal (calculated from /goal:Point and /odom:Odometry), and the robot's current twist (linear and angular velocity from /odom:Odometry) as inputs. It then outputs the twist command (/cmd_vel:Twist) for the controller.

This project is a part of the course FRA532 Mobile Robots at Institute of Field Robotics, King Mongkut's University of Technology Thonburi. 

## Requirements
- Ubuntu 20.04
- ROS2 humble

## Related Libraries
1. **Stage simulator** is use to create a lighweight simulation to efficiently train the reinforcement learning agent.
2. **Gymnasium** provide tools to contructs reinforcement learning framework such as gymnasium.Env that can be used to defined action space, observation space, and reward function.
3. **Stable Baseline 3** is use to calculate the gradient and update policy network and value network of the Soft Actor-Critic (SAC) algorithm.

## Installation

1. Clone the repository to your /src folder in your workspace.

```bash
git clone https://github.com/WasupolHengsritawat/rl_base_navigation.git
cd ~/<your_ws>
colcon build
```
2. Follow step 2 to 5 from [Lecture 5 : SLAM and Navigation](https://github.com/kittinook/MobileRobotics2025/tree/navigation-sol) to install the MiR Robot Package and Warehouse World Package.

3. Install Stage_ros2 for 2D simulation. You can visit [tuw-robotics/stage_ros2 repository](https://github.com/tuw-robotics/stage_ros2) for more information about the simulator.
```bash
sudo apt-get install git cmake g++ libjpeg8-dev libpng-dev libglu1-mesa-dev libltdl-dev libfltk1.1-dev
cd ~/<your_ws>
mkdir src
cd src
git clone --branch ros2 git@github.com:tuw-robotics/Stage.git
git clone --branch humble git@github.com:tuw-robotics/stage_ros2.git
cd ~/<your_ws>
colcon build --symlink-install --cmake-args -DOpenGL_GL_PREFERENCE=LEGACY
colcon build --symlink-install --packages-select stage_ros2     
```

4. Install reinforcement learning related libraries.
```bash
pip install stable-baselines3
pip install gymnasium
```

5. Build and source your workspace
```bash
cd ~/<your_ws>
colcon build && source install/setup.bash
```
## Networks Structures
The Soft Actor-Critic (SAC) is used in this work due to its high exploration rate via entropy regularization (SAC maximizes both the reward and the entropy of the action). This is particularly suitable for obstacle avoidance, which is a problem with sparse rewards (penalizing the agent only when it collides with objects and rewarding it for moving in the right direction).

![](https://github.com/WasupolHengsritawat/rl_base_navigation/blob/main/media/PolicyNetworkStructure.png)

The policy network structure follows this [literature](https://journals.sagepub.com/doi/10.1177/0278364920916531). It processes three timesteps of 360-range LiDAR data using two 1D convolutional layers: the first with 32 kernels of size 4 and stride 2, and the second with 32 kernels of size 3 and stride 2. The output is then passed to a fully connected layer with 180 nodes.

The output from the first fully connected layer is concatenated with the distance to the goal and the current twist and then passed through two additional fully connected layers, each with 128 units. The final output consists of the mean and standard deviation of the twist command distribution, from which an action is sampled for the agent.

The structure of the value network is similar to that of the policy network but includes the agent's action as an additional feature in the second fully connected layer, along with the extracted features from LiDAR data, distance to the goal, and current twist.

## Reward Function
The reward consists three terms:
- **Right Direction Term** – If the agent reaches the goal (within 0.2 m of the goal), it receives a reward of **10**. Otherwise, it receives a reward equal to the difference between the distance to the goal in the previous step and the distance to the goal in the current step.

- **Collision Term** – If the agent collides with an obstacle (determined when the minimum LiDAR reading is less than 0.65 m), it receives a reward of **-10**.

- **High Rotation Term** – The agent receives a reward of **-0.1 × |w*|**, where **w** is the current angular velocity, if **|w| > 0.6** rad/s. This penalty discourages excessive spinning, as high angular velocity makes it difficult to control the agent's direction.


## Training Policy Networks

Since Gazebo is a physics simulator, training reinforcement learning agents in Gazebo requires significant computational power, and its complexity may affect learning convergence. Instead, the agents (i.e., the local planners) are trained in a lightweight simulation such as Stage.

This repository includes two types of obstacle avoidance training: simple and random.

-  **Simple environment training** is more computationally efficient than random training. The environment remains fixed for each iteration, but the dynamic obstacle velocities are randomized. To train or modify the agent in the simple environment, run the following Python script:
    ```bash
    /usr/bin/python3 ~/<your_ws>/src/rl_base_navigation/scripts/stage_simple_gym_train.py
    ```

- **Random environment training** regenerates a new Stage simulation for every episode, meaning that static objects differ between episodes. To train or modify the agent in the random environment, run the following Python script:
    ```bash
    /usr/bin/python3 ~/<your_ws>/src/rl_base_navigation/scripts/random_simple_gym_train.py
    ```

You can visualize the trained model by selecting the model using: `MODEL_NAME = <model_name>` in the following script:
```bash
code ~/<your_ws>/src/rl_base_navigation/scripts/stage_gym_play.py
```
To view the reward and episode length logs, use:
```bash
tensorboard --logdir=tensorboard_logs --bind_all
```
Model and log examples can be found in the *models* and *tensorboard_logs* directories, respectively. It is recommended to move these files out of the source directory (to your workspace) if you want to visualize the model using the above commands.

## Usage
1. Run a simulation with MiR100 robot. For example, run
```bash
# In your workspace
source install/setup.bash
ros2 launch fra532_gazebo sim.launch.py
```
2. Modify `MODEL_NAME = <model_name>` to the model of your choices in
```bash
code ~/<your_ws>/src/rl_base_navigation/scripts/gazebo_simple_gym_play.py
```
Then, run it using
```bash
/usr/bin/python3 ~/<your_ws>/src/rl_base_navigation/scripts/gazebo_simple_gym_play.py
```

## Demo
https://github.com/WasupolHengsritawat/rl_base_navigation/blob/main/media/demo.mp4

## Results
### Learning Results
![](https://github.com/WasupolHengsritawat/rl_base_navigation/blob/main/media/ep_len_and_ep_rew.png)

After training for 50,000 timesteps, SAC1 and SAC3 were trained in the random environment, while SAC4 was trained in the simple environment. None of the three models achieved an acceptable level of reward or reward convergence. This may be due to insufficient fine-tuning of the model and limited training time.

However, the models trained in the random environment tended to achieve higher rewards. This could be due to their greater generalization ability, but a definitive conclusion cannot be drawn due to the lack of sufficient data. More models with improved training and fine-tuning are required for a thorough statistical analysis.

### Implementation Results
https://github.com/WasupolHengsritawat/rl_base_navigation/blob/main/media/stage.webm
https://github.com/WasupolHengsritawat/rl_base_navigation/blob/main/media/gazebo.webm

The agent controlled by the model are having some sense of object avoidance but stills perform really bad. Better reward tuning will improve the agent behavior.

## Author
- Wasupol Hengsritawat 64340500049


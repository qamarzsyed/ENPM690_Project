# ENPM 690 Final Project : Model-free RL to optimize lap times
Qamar Syed

### Overview
Code to train a PPO model in a modified car racer v2 environment and deploy the model in ROS 2 gazebo  

Contained in this directory:  
ROSFiles
- rc_car : ROS2 package to deploy in track gazebo world  

TrainingFiles
- BigL : directory containing training checkpoints and saved policies/models  
- car_racing.py : modified gymnasium environment  
- raylib_ppo.py : training code for ppo  
- deploy_raylib.py : code to deploy model in gymnasium  


### Requirements
gymnasium[box2d]  
ray[rllib]==2.4.0  
stable-baselines3  
ROS2 Galactic & ROS2 Python Packages

### Training instructions
Before running any of the training files, insert the provided modified car_racing.py into your gymnasium package to replace the original one.

Enter the following commands into the terminal to run the training algorithm
1. 'cd TrainingFiles'
2. 'python raylib_ppo.py'

Enter the following commands to deploy the training algorithm in gymnasium after editing the model path in deploy_raylib.py
1. 'cd TrainingFiles'
2. 'python deploy_raylib.py'

### ROS 2 Deployment instructions
Before running, ensure that model_path in rl_controller.py turns to your desired model checkpoint.

Steps:
1. Place the 'rc_car' ROS2 package into a ROS2 workspace
2. Run 'colcon build' at the workspace upper level to build the package
3. Run 'ros2 launch rc_car competition.launch.py' to launch the robot in the track world
4. Run 'ros2 run rc_car rl_controller.py' to deploy the trained model in the ROS2 simulation

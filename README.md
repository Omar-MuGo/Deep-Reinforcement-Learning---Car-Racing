# Car Racing v2 Solver


## Overview
This repository contains a solution to the OpenAI Gym environment "Racing Car v2". The goal of this project is to implement an agent capable of efficiently navigating a race track in the OpenAI Gym simulation environment.

![Racing Car v1](https://www.gymlibrary.dev/_images/car_racing.gif)

Source: https://www.gymlibrary.dev/environments/box2d/car_racing/


## Environment Description
The Racing Car v2 environment simulates a car racing track. The agent controls a car and must navigate through the track while avoiding obstacles and completing the course as quickly as possible.

|        Category       |               Description               |
|-----------------------|----------------------------------|
|   **Action Space**    | 5 actions: do nothing, steer left, steer right, gas, brake |
| **Observation Shape** |            (96, 96, 3) RGB Image           |


## Solution Approach
This project utilizes reinforcement learning techniques, specifically a Deep Q Network, to train an agent to navigate the race track effectively. We employ a convolutional neural network (CNN) as the function approximator to estimate action-values. The agent follows an epsilon-greedy policy.

We consider the environment is solved after the mean return over a window of 100 episodes is above 780 (which approximately corresponds to a full lap).


## Installation
1. Clone the repository:
```
git clone https://github.com/your_username/racing-car-v1-solver.git
```
2. Install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
3. Install the dependencies:
```
pip install -r requirements.txt
```

## Usage
1. To train the agent run: **train.ipynb**
2. To evaluate the trained agent run: **test.ipynb**

## Results
The plot below shows the reward obtained throughout all the episodes the agent used as training. We observe an increasing trend in the episode reward as we increase the number of episodes, showing that the agent is learning.

Below, it is possible to see the agent's progress throghout the training process.

![Training](https://raw.githubusercontent.com/Omar-MuGo/Deep-Reinforcement-Learning---Car-Racing/master/img/agent_reward.png)


Open in YouTube:

[![Racing Car v1](https://raw.githubusercontent.com/Omar-MuGo/Deep-Reinforcement-Learning---Car-Racing/master/img/training.gif)](https://youtu.be/C4P4RTBjbjs)


## Future Work
- Fine-tuning convolutional neural network architecture to potentially improve performance. Consider adding additional inputs to the network e.g. the action taken in the previous timestep.
- Implement improvements to the DQN architecture such as dueling networks and prioritized experience replay.

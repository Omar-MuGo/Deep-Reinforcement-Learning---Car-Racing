"""
Filename: model.py
Author: Omar Rodrigo Muñoz Gómez
Created: February 27, 2024
Description: Definition of the QNetwork architecture using a custom CNN
             as the funciton approximator to estimate action-values.
Acknowledgements: This code is based on the DQN code from the Reinforcement Learning Udacity Nanodegree 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_channels, state_height, state_width, action_size, seed, conv1_channels=32, conv2_channels=64, conv3_channels=64, fc1_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_channels (int): Number of channels in the state image
            state_height (int): Height of the state image
            state_width (int): Width of the state image
            action_size (int): Dimension of each action
            seed (int): Random seed
            conv1_channels (int): Number of output channels for the first convolutional layer
            conv2_channels (int): Number of output channels for the second convolutional layer
            conv3_channels (int): Number of output channels for the third convolutional layer
            fc1_units (int): Number of units in the first fully connected layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(state_channels, conv1_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(conv2_channels, conv3_channels, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(conv3_channels * state_height * state_width, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        # Convert the RGB image to (pseudo) grayscale
        state = torch.mean(state, dim=3, keepdim=True)
        
        # Flip channels to match the image format for the network
        state = state.permute(0, 3, 1, 2)
        
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the output from convolutional layers
        x = F.relu(self.fc1(x))
        return self.fc2(x)
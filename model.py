import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class BeaconVision(nn.Module):
    def __init__(self, beacon_size):
        super(BeaconVision, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 6)

        # policy
        self.pi_1 = nn.Linear(2+6, 32)
        self.pi_2 = nn.Linear(32, 32)
        self.pi_mean = nn.Linear(32, 2)
        self.pi_std = nn.Linear(32, 2)

        # predictor
        self.b_1 = nn.Linear(6, 16)
        self.b_2 = nn.Linear(16, 16)
        self.b_mean = nn.Linear(16, beacon_size)
        self.b_std = nn.Linear(16, beacon_size)

        # other stuff
        self.apply(weights_init_)
        self.mse_func = nn.MSELoss()

    def feature_encoder(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def policy(self, state, phi):
        x = torch.cat((state, phi), 1)
        x = torch.tanh(self.pi_1(x))
        x = torch.relu(self.pi_2(x))
        x_mean = self.pi_mean(x)
        x_std = torch.exp(0.5 * self.pi_std(x))
        eps = torch.randn_like(x_std)
        x = x_mean + x_std * eps
        return x

    def predictor(self, phi):
        x = torch.tanh(self.b_1(phi))
        x = torch.tanh(self.b_2(x))
        x_mean = self.b_mean(x)
        x_std = torch.exp(0.5 * self.b_std(x))
        eps = torch.randn_like(x_std)
        x = x_mean + x_std * eps
        return x

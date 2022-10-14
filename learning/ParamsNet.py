import torch
from torch import nn
import numpy as np


class ParamsNet(nn.Module):
    def __init__(self, device):
        super(ParamsNet, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(64*64*3, 64*64)
        self.fc2 = nn.Linear(64*64, 32*32)
        self.fc3 = nn.Linear(32 * 32, 16 * 16)
        self.fc4 = nn.Linear(16 * 16, 4*4)
        self.fc5 = nn.Linear(4*4, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x


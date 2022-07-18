import torch
from torch import nn, optim, cuda, Tensor
import numpy as np


class ParamsNet(nn.Module):
    def __init__(self):
        super(ParamsNet, self).__init__()
        self.fc1 = nn.Linear(64*64*3, 64*64)
        self.fc2 = nn.Linear(64*64, 32*32)
        self.fc3 = nn.Linear(32 * 32, 16 * 16)
        self.fc4 = nn.Linear(16 * 16, 4*4)
        self.fc5 = nn.Linear(4*4, 2)
        self.relu = nn.ReLU()
        self.device = torch.device('cuda:0')
        torch.cuda.set_device(self.device)
        self.to(self.device)
        self.eval()

    def forward(self, x):
        x = x.astype(np.float32)
        if len(x.shape) != 4:
            x = np.array([x])
        x = torch.from_numpy(x).to(self.device)
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

    def save(self, name):
        torch.save(self.state_dict(), f'{name}')

    def load(self, path):
        self.load_state_dict(torch.load(path))


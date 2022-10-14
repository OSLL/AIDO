import torch
from torch import nn


class ParallelNet(nn.Module):
    def __init__(self, device):
        super(ParallelNet, self).__init__()
        self.device = device
        self.d_conv = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5),
            nn.BatchNorm2d(12),

            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(12, 36, kernel_size=5),
            nn.BatchNorm2d(36),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(36, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.phi_conv = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5),
            nn.BatchNorm2d(12),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(12, 36, kernel_size=5),
            nn.BatchNorm2d(36),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(36, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.d_decoder = nn.Sequential(
            nn.Linear(48*4*4, 48),
            nn.ReLU(),
            nn.Linear(48, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        self.phi_decoder = nn.Sequential(
            nn.Linear(48 * 4 * 4, 48),
            nn.ReLU(),
            nn.Linear(48, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        d = self.d_conv(x)
        phi = self.phi_conv(x)
        d = d.view(d.size(0), d.size(1) * d.size(2) * d.size(3))
        phi = phi.view(phi.size(0), phi.size(1) * phi.size(2) * phi.size(3))
        d = self.d_decoder(d)
        phi = self.phi_decoder(phi)
        return d, phi


class ParallelLinearNet(nn.Module):
    def __init__(self, device):
        super(ParallelLinearNet, self).__init__()
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5),
            nn.BatchNorm2d(12),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(12, 36, kernel_size=5),
            nn.BatchNorm2d(36),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(36, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.d_decoder = nn.Sequential(
            nn.Linear(48 * 4 * 4, 48),
            nn.ReLU(),
            nn.Linear(48, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        self.phi_decoder = nn.Sequential(
            nn.Linear(48 * 4 * 4, 48),
            nn.ReLU(),
            nn.Linear(48, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        x = self.conv(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        d = self.d_decoder(x)
        phi = self.phi_decoder(x)
        return d, phi


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
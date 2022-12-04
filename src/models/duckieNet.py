import torch.nn as nn


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DuckieNet(nn.Module):
    def __init__(self):
        super(DuckieNet, self).__init__()
        self.main = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2)
        )
        self.branch1_1 = nn.Sequential(
            BasicConv2d(64, 128, kernel_size=3, stride=2),
            BasicConv2d(128, 256, kernel_size=3, stride=1),
            BasicConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        )
        self.branch1_2 = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.branch2_1 = nn.Sequential(
            BasicConv2d(64, 128, kernel_size=3, stride=2),
            BasicConv2d(128, 256, kernel_size=3, stride=1),
            BasicConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        )
        self.branch2_2 = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.main(x)
        d = self.branch1_1(x)
        phi = self.branch2_1(x)
        d = d.view(d.size(0), d.size(1) * d.size(2) * d.size(3))
        phi = phi.view(phi.size(0), phi.size(1) * phi.size(2) * phi.size(3))
        d = self.branch1_2(d)
        phi = self.branch2_2(phi)
        return d, phi

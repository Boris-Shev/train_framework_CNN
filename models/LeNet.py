from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.batchnorm import BatchNorm1d
import wandb

class LeNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(num_channels, 6, 5) # in: num_channels*28*28   out: 6*24*24
        self.bn2d1 = BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2)    # in: 6*24*24   out: 6*12*12

        self.conv2 = nn.Conv2d(6, 16, 3)# in: 6*12*12   out: 16*10*10
        self.bn2d2 = BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)    # in: 16*10*10  out: 16*5*5

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.bn1d1 = BatchNorm1d(120)

        self.fc2 = nn.Linear(120, 84)
        self.bn1d2 = BatchNorm1d(84)

        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn2d1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2d2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.bn1d1(x)

        x = F.relu(self.fc2(x))
        x = self.bn1d2(x)

        x = self.fc3(x)

        return x
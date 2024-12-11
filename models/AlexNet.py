from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.batchnorm import BatchNorm1d
import wandb

class AlexNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 96, 11, stride=4), # in: num_channels*224*224   out: 96*54*54
            BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2)    # in: 96*54*54   out: 96*26*26
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, padding=2), # in: 96*26*26   out: 256*26*26
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2)    # in: 256*26*26  out: 256*12*12
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding=1), # in: 256*12*12   out: 384*12*12
            nn.ReLU(inplace=True),
            BatchNorm2d(384)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, padding=1), # in: 384*12*12   out: 384*12*12
            nn.ReLU(inplace=True),  
            BatchNorm2d(384)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1), # in: 384*12*12   out: 256*12*12
            nn.ReLU(inplace=True),  
            BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2)    # in: 256*12*12  out: 256*5*5
        )

        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*5*5, 4096),
            nn.ReLU(inplace=True),
            BatchNorm1d(4096),
            nn.Dropout(0.5)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            BatchNorm1d(4096),
            nn.Dropout(0.5)
        )

        self.linear3 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.features = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )

        self.classifier = nn.Sequential(
            self.linear1,
            self.linear2,
            self.linear3
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
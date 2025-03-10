import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        # Adjust residual size if stride is larger than 1
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += res
        out = self.relu(out)
        return out

def getResnetSequence(in_channel, out_channel, stride):
    layers = nn.Sequential(
        ResNetBlock(in_channel, out_channel, stride)
    )
    return layers

class ResNetRegression(nn.Module):
    
    def __init__(self):
        super(ResNetRegression, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (7,7), padding=3, stride=2)
        self.maxPool = nn.MaxPool2d((3,3), stride=2, padding=1)
        self.relu = nn.ReLU()
        self.conv2_x = getResnetSequence(64, 64, 1)
        self.conv3_x = getResnetSequence(64, 128, 1)
        self.conv4_x = getResnetSequence(128, 256, 1)
        self.conv5_x = getResnetSequence(256, 512, 2)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxPool(out)
        out = self.relu(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avgPool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = out.squeeze(1)
        return out
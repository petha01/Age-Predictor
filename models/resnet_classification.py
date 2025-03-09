import torch
import torch.nn as nn

def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

class ResNetBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, k_size, stride, padding):
        super(ResNetBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channel, out_channel, (k_size, k_size), padding=padding, stride=stride)
        nn.init.kaiming_normal_(self.conv_1.weight)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, (k_size, k_size), padding=padding, stride=1)
        nn.init.kaiming_normal_(self.conv_1.weight)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResNetClassification(nn.Module):
    
    def __init__(self, num_class):
        super(ResNetClassification, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, (7,7), padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3,3), stride=2, padding=1),
            ResNetBlock(64, 64, 3, 1, 1),
            ResNetBlock(64, 128, 3, 1, 1),
            ResNetBlock(128, 256, 3, 2, 1),
            ResNetBlock(256, 512, 3, 1, 1),
            nn.AvgPool2d((4,4)),
            Flatten(),
            nn.Linear(512, num_class)
        )
        
    def forward(self, x):
        return self.model(x)
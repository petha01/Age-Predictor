import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Add dropout
        self.dropout = nn.Dropout(0.25)
        
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
        out = self.dropout(out)
        return out

def getResnetSequence(in_channel, out_channel, stride):
    layers = nn.Sequential(
        ResNetBlock(in_channel, out_channel, stride)
    )
    return layers

class ResNetDropout(nn.Module):
    
    def __init__(self, num_class):
        super(ResNetDropout, self).__init__()
        pretrained_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.conv1 = nn.Conv2d(3, 64, (7,7), padding=3, stride=2)
        self.conv1.weight.data = pretrained_resnet.conv1.weight.data.clone()

        self.maxPool = nn.MaxPool2d((3,3), stride=2, padding=1)
        self.relu = nn.ReLU()

        self.conv2_x = getResnetSequence(64, 64, 1)
        self.conv2_x[0].conv1.weight.data = pretrained_resnet.layer1[0].conv1.weight.data.clone()
        self.conv2_x[0].conv2.weight.data = pretrained_resnet.layer1[0].conv2.weight.data.clone()
        self.conv2_x[0].bn1.load_state_dict(pretrained_resnet.layer1[0].bn1.state_dict())
        self.conv2_x[0].bn1.load_state_dict(pretrained_resnet.layer1[0].bn1.state_dict())

        self.conv3_x = getResnetSequence(64, 128, 1)
        self.conv3_x[0].conv1.weight.data = pretrained_resnet.layer2[0].conv1.weight.data.clone()
        self.conv3_x[0].conv2.weight.data = pretrained_resnet.layer2[0].conv2.weight.data.clone()
        self.conv3_x[0].bn1.load_state_dict(pretrained_resnet.layer2[0].bn1.state_dict())
        self.conv3_x[0].bn1.load_state_dict(pretrained_resnet.layer2[0].bn1.state_dict())

        self.conv4_x = getResnetSequence(128, 256, 1)
        self.conv4_x[0].conv1.weight.data = pretrained_resnet.layer3[0].conv1.weight.data.clone()
        self.conv4_x[0].conv2.weight.data = pretrained_resnet.layer3[0].conv2.weight.data.clone()
        self.conv4_x[0].bn1.load_state_dict(pretrained_resnet.layer3[0].bn1.state_dict())
        self.conv4_x[0].bn1.load_state_dict(pretrained_resnet.layer3[0].bn1.state_dict())

        self.conv5_x = getResnetSequence(256, 512, 2)
        self.conv5_x[0].conv1.weight.data = pretrained_resnet.layer4[0].conv1.weight.data.clone()
        self.conv5_x[0].conv2.weight.data = pretrained_resnet.layer4[0].conv2.weight.data.clone()
        self.conv5_x[0].bn1.load_state_dict(pretrained_resnet.layer4[0].bn1.state_dict())
        self.conv5_x[0].bn1.load_state_dict(pretrained_resnet.layer4[0].bn1.state_dict())

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_class)        

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
        return out
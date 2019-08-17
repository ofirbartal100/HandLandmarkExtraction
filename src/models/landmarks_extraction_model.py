import torch
import torch.nn as nn
import torchvision

# taken from torchvision.resnet.py
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LandmarksExtractionModel(torchvision.models.ResNet):
    def __init__(self, saved_model_file_path=None):
        super().__init__(Bottleneck, [3, 4, 6, 3])
        self.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3))
        self.avgpool = torch.nn.AvgPool2d(2)
        self.fc = torch.nn.Linear(2048, 42)

        if saved_model_file_path:
            self.load_state_dict(torch.load(saved_model_file_path))

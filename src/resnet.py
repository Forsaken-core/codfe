import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet as TorchResNet


class ResNet(TorchResNet):
    def __init__(self, in_channels=39, out_channels=2):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=1)
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.fc_reg = nn.Linear(512, out_channels)
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, images):
        x = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        # Shared features feed both heads
        features = self.fc_reg(x)
        return self.fc(features), features

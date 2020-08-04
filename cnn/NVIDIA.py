import torch
import torch.nn as nn


# TODO: normalization
class NVIDIA(nn.Module):
    """https://developer.nvidia.com/blog/deep-learning-self-driving-cars/"""

    def __init__(self, num_channels):
        super(NVIDIA, self).__init__()
        self.conv1 = conv(num_channels, 24, kernel_size=5, stride=2)
        self.conv2 = conv(24, 36, kernel_size=5, stride=2)
        self.conv3 = conv(36, 48, kernel_size=5, stride=2)
        self.drop = nn.Dropout2d(p=0.5)
        self.conv4 = conv(48, 64, kernel_size=3, stride=1)
        self.conv5 = conv(64, 64, kernel_size=3, stride=1)
        self.linear1 = linear(64*1*20, 100)
        self.linear2 = linear(100, 50)
        self.linear3 = linear(50, 10)
        self.linear4 = nn.Linear(10, 1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode='fan_in')
                if layer.bias is not None:
                    layer.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.drop(self.conv4(x))
        x = self.conv5(x)
        x = x.view(-1, 64*1*20)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x.squeeze(1)


def conv(in_planes, out_planes, kernel_size=5, stride=1, padding=0, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True),
            nn.ELU())


def linear(in_features, out_features):
    return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ELU())


if __name__ == '__main__':
    understand_tensor()

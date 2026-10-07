import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A conventional Convolutional Neural Network.
    Features: Sequential Conv -> ReLU -> MaxPool -> FC layers.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Input: 3x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 3 pools, size is 32 -> 16 -> 8 -> 4. 128 channels * 4 * 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, return_features=False):
        f1 = self.pool(F.relu(self.conv1(x)))
        f2 = self.pool(F.relu(self.conv2(f1)))
        f3 = self.pool(F.relu(self.conv3(f2)))
        
        out = f3.view(-1, 128 * 4 * 4)
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)
        
        if return_features:
            return out, f3
        return out

class DenseLayer(nn.Module):
    """
    A single layer inside a Dense Block.
    Applies BN -> ReLU -> Conv2d.
    """
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        # Feature reuse: concatenate input with output
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    """
    A block of dense layers.
    Each layer gets the concatenated output of all preceding layers in the block.
    """
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    """
    Connects Dense Blocks and reduces dimensions.
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))

class MiniDenseNet(nn.Module):
    """
    A small DenseNet implementation for educational purposes.
    """
    def __init__(self, num_classes=10, growth_rate=12, block_layers=[3, 3, 3]):
        super(MiniDenseNet, self).__init__()
        
        num_channels = 24
        # Initial Convolution
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        
        self.dense1 = DenseBlock(block_layers[0], num_channels, growth_rate)
        num_channels += block_layers[0] * growth_rate
        self.trans1 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense2 = DenseBlock(block_layers[1], num_channels, growth_rate)
        num_channels += block_layers[1] * growth_rate
        self.trans2 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense3 = DenseBlock(block_layers[2], num_channels, growth_rate)
        num_channels += block_layers[2] * growth_rate
        
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x, return_features=False):
        out = self.conv1(x)
        
        out = self.dense1(out)
        out = self.trans1(out)
        
        out = self.dense2(out)
        out = self.trans2(out)
        
        out = self.dense3(out)
        features = F.relu(self.bn_final(out))
        
        out = self.pool(features)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        if return_features:
            return out, features
        return out

import torch
import torch.nn as nn


class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        # First Convolution-BatchNorm-ReLU
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Second Convolution-BatchNorm
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        # ReLU Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        # Apply first set of Conv-BN-ReLU
        out = self.relu(self.bn1(self.conv1(x)))
        
        # Apply second set of Conv-BN
        out = self.bn2(self.conv2(out))
        
        # Add input (residual connection) and apply final ReLU
        out = self.relu(out + x)
        
        return out


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        # (a) Initial Convolution
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        # (b) Batch Normalization
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # (c) ReLU Activation
        self.relu = nn.ReLU(inplace=True)
        
        # (d) Max Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # (e) Residual Block
        self.block = Block(num_channels)
        
        # (f) Adaptive Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # (g) Linear Layer
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor from (N, C, 1, 1) to (N, C)
        x = self.fc(x)
        
        return x
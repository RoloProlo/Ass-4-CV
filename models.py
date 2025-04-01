import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallObjectDetector(nn.Module):
    def __init__(self):
        super(SmallObjectDetector, self).__init__()

        # Convolutional backbone (input 112x112)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 112 → 56

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 56 → 28

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # 28 → 14

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 64 kernels
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)  # 14 → 7

        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 32 kernels
        self.bn5 = nn.BatchNorm2d(32)

        # Flatten + FC Layers
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(7 * 7 * 32, 512)
        self.fc2 = nn.Linear(512, 343)

    def forward_conv_layers(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        return x

    def forward(self, x):
        x = self.forward_conv_layers(x)          # [B, 32, 7, 7]
        x = x.view(x.size(0), -1)                # Flatten to [B, 1568]
        x = F.relu(self.fc1(x))                  # FC(512)
        x = self.dropout(x)                      # Dropout
        x = torch.sigmoid(self.fc2(x))           # Output layer + sigmoid
        return x

################################################################
#CODE FOR CHOICE TASK 1
################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.mish(out)


class CHOICE1(nn.Module):
    def __init__(self):
        super(CHOICE1, self).__init__()

        # Convolutional Backbone with Residual Connections
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.res1 = ResidualBlock(16, 32, stride=2)
        self.res2 = ResidualBlock(32, 64, stride=2)
        self.res3 = ResidualBlock(64, 64, stride=2)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Feature Pyramid Fusion
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_fusion = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_fusion = nn.BatchNorm2d(32)

        # Global Average Pooling Instead of Fully Connected Layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 343)

    def forward(self, x):
        x = F.mish(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = F.mish(self.bn2(self.conv2(x)))

        # Multi-Scale Feature Fusion
        x = self.upsample(x)
        x = F.mish(self.bn_fusion(self.conv_fusion(x)))

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # No sigmoid, use BCEWithLogitsLoss
        return x


if __name__ == "__main__":
    model = SmallObjectDetector()
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")












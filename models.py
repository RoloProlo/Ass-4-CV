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

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # ✅ as per spec: 64 kernels
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)  # 14 → 7

        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # ✅ as per spec: 32 kernels
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



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class CHOICE1(nn.Module):
    def __init__(self):
        super(CHOICE1, self).__init__()

        # Convolutional Backbone with Residual Connections
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # Strided conv instead of max pooling
        self.bn1 = nn.BatchNorm2d(16)

        self.res1 = ResidualBlock(16, 32, stride=2)
        self.res2 = ResidualBlock(32, 64, stride=2)
        self.res3 = ResidualBlock(64, 64, stride=2)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2)  # Dilated conv for larger receptive field
        self.bn2 = nn.BatchNorm2d(32)

        # Global Average Pooling instead of Fully Connected Layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 343)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x


if __name__ == "__main__":
    model = SmallObjectDetector()
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")













# class SmallObjectDetector(nn.Module):
#     def __init__(self):
#         super(SmallObjectDetector, self).__init__()
#
#         # Convolutional backbone (input 112x112)
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.pool1 = nn.MaxPool2d(2, 2)  # 112 → 56
#
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool2 = nn.MaxPool2d(2, 2)  # 56 → 28
#
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.pool3 = nn.MaxPool2d(2, 2)  # 28 → 14
#
#         self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.pool4 = nn.MaxPool2d(2, 2)  # 14 → 7
#
#         # Final conv before prediction head
#         self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm2d(64)
#
#         # Head: 7 outputs per grid cell (4 box, 1 obj, 2 class)
#         self.head = nn.Conv2d(64, 7, kernel_size=1)  # preserves 7x7
#
#     def forward_conv_layers(self, x):
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool2(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool3(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool4(F.relu(self.bn4(self.conv4(x))))
#         x = F.relu(self.bn5(self.conv5(x)))
#         return x
#
#     def forward(self, x):
#         x = self.forward_conv_layers(x)     # [B, 64, 7, 7]
#         x = self.head(x)                    # [B, 7, 7, 7]
#         x = x.permute(0, 2, 3, 1)           # [B, 7, 7, 7] for YOLO format
#         return x



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class SmallObjectDetector(nn.Module):
#     def __init__(self):
#         super(SmallObjectDetector, self).__init__()
#
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.bn6 = nn.BatchNorm2d(64)
#
#         # Pooling layer (2x3 kernel, stride 2)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # Batch normalization layers
#         self.bn1 = nn.BatchNorm2d(16)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.bn5 = nn.BatchNorm2d(32)
#
#         # Dropout layer
#         self.dropout = nn.Dropout(0.4)
#
#         # To dynamically calculate the input size for the fully connected layer
#         self._to_linear = None
#         self._calculate_linear_input_size()
#
#         # Fully connected layers
#         self.fc1 = nn.Linear(self._to_linear, 512)
#         self.fc2 = nn.Linear(512, 343)  # Output layer
#
#     def _calculate_linear_input_size(self):
#         # Pass a dummy tensor through the convolutional and pooling layers to calculate the output size
#         dummy_input = torch.zeros(1, 3, 112, 112)  # Assuming input size of 112x112 with 3 channels
#         dummy_output = self.forward_conv_layers(dummy_input)
#         self._to_linear = dummy_output.view(1, -1).size(1)
#
#     def forward_conv_layers(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> Pool1
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> Pool2
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> Pool3
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv4 -> Pool4
#         x = F.relu(self.bn5(self.conv5(x)))  # Conv5 (no pooling after last conv)
#         x = self.pool(F.relu(self.bn6(self.conv6(x))))  # New downsampling step
#
#         return x
#
#     def forward(self, x):
#         # Pass through the convolutional layers
#         x = self.forward_conv_layers(x)
#
#         # Flatten before fully connected layers
#         x = torch.flatten(x, start_dim=1)
#
#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)  # Apply dropout
#         x = torch.sigmoid(self.fc2(x))  # ✅ correct place for sigmoid
#
#         print("Final feature map:", x.shape)
#
#         return x
#
#
# if __name__ == "__main__":
#     # Instantiate the model
#     model = SmallObjectDetector()
#
#     # Print model summary
#     print(model)
#
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total trainable parameters: {total_params:,}")
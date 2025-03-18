import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallObjectDetector(nn.Module):
    def __init__(self):
        super(SmallObjectDetector, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Pooling layers (2x3, stride 2, no padding)
        self.pool = nn.MaxPool2d(kernel_size=(2,3), stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 6 * 6, 512)  # Adjusted input size after pooling
        self.fc2 = nn.Linear(512, 343)  # Output layer

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

        # Dropout
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> Pool1
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> Pool2
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> Pool3
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv4 -> Pool4
        x = F.relu(self.bn5(self.conv5(x)))  # Conv5 (no pooling after last conv)

        # Flatten before FC layers
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Output layer

        return x

# Instantiate the model
model = SmallObjectDetector()

# Print model summary
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

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

        # Pooling layer (2x3 kernel, stride 2)
        self.pool = nn.MaxPool2d(kernel_size=(2, 3), stride=2, padding=0)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

        # Dropout layer
        self.dropout = nn.Dropout(0.4)

        # To dynamically calculate the input size for the fully connected layer
        self._to_linear = None
        self._calculate_linear_input_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 343)  # Output layer

    def _calculate_linear_input_size(self):
        # Pass a dummy tensor through the convolutional and pooling layers to calculate the output size
        dummy_input = torch.zeros(1, 3, 112, 112)  # Assuming input size of 112x112 with 3 channels
        dummy_output = self.forward_conv_layers(dummy_input)
        self._to_linear = dummy_output.view(1, -1).size(1)

    def forward_conv_layers(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> Pool1
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> Pool2
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> Pool3
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv4 -> Pool4
        x = F.relu(self.bn5(self.conv5(x)))  # Conv5 (no pooling after last conv)
        return x

    def forward(self, x):
        # Pass through the convolutional layers
        x = self.forward_conv_layers(x)

        # Flatten before fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Output layer

        return x


if __name__ == "__main__":
    # Instantiate the model
    model = SmallObjectDetector()

    # Print model summary
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

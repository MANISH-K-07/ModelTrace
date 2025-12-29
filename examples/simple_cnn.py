import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)

        # For input 8x8:
        # 8x8 → conv1 → 6x6
        # 6x6 → conv2 → 4x4
        # Channels = 16 → 16 * 4 * 4 = 256
        self.fc = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

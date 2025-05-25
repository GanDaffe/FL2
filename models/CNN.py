from torch import nn
import torch

class CNN(nn.Module):
    def __init__(self, in_channel, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=128,
            kernel_size=5,
            padding='same')

        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=5,
            padding='same')

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding='same')

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding='same')

        self.conv5 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding='same')

        self.conv6 = nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            padding='same')

        self.conv7 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding='same')

        self.conv8 = nn.Conv2d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            padding='same')

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.lin1 = nn.Linear(in_features=8 * 1 * 16, out_features=256)
        self.classification = nn.Linear(256, out_features=num_classes)

    def forward(self, X):
        X = self.pool(self.relu(self.conv2(self.relu(self.conv1(X)))))
        X = self.pool(self.relu(self.conv4(self.relu(self.conv3(X)))))
        X = self.pool(self.relu(self.conv6(self.relu(self.conv5(X)))))
        X = self.pool(self.relu(self.conv8(self.relu(self.conv7(X)))))

        X = torch.flatten(X, start_dim=1)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.classification(X)

        return X
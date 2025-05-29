from torch import nn  
import torch.nn.functional as F
import torch

class CNN_header(nn.Module):
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

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=16)
      
        self.lin1 = nn.Linear(16 * 2 * 32, 256)

    def forward(self, X):
        X = self.pool(self.bn1(self.relu(self.conv2(self.relu(self.conv1(X))))))
        X = self.pool(self.bn2(self.relu(self.conv4(self.relu(self.conv3(X))))))
        X = self.pool(self.bn3(self.relu(self.conv6(self.relu(self.conv5(X))))))

        X = torch.flatten(X, start_dim=1) 
        X = self.lin1(X)
        X = self.relu(X)
        X = self.dropout(X)

        return X

    
class ModelMoon(nn.Module): 

    def __init__(self):
        super(ModelMoon, self).__init__() 

      
        self.features = CNN_header(1, 3)
        num_ftrs = 256
        # self.ln1 = nn.Linear(num_ftrs, num_ftrs) 
        # self.ln2 = nn.Linear(num_ftrs, out_dim) 
        self.l3 = nn.Linear(num_ftrs, 3) 

    def forward(self, x): 
        h = self.features(x) 
        h = h.view(h.size(0), -1)

        # x = self.ln1(h) 
        # x = F.relu(x)
        # x = self.l2(x) 
        
        y = self.l3(h) 
        return h, h, y

def init_model(): 
    return ModelMoon()

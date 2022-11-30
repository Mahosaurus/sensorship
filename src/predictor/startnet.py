import torch.nn as nn
import torch.nn.functional as F

class StartNet(nn.Module):

    def __init__(self):
        super(StartNet, self).__init__()
        self.fc1 = nn.Linear(3, 9)
        self.fc2 = nn.Linear(9, 1)            

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

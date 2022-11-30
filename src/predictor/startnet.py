import torch.nn as nn

class StartNet(nn.Module):

    def __init__(self):
        super(StartNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 15),
            nn.ReLU(),
            nn.Linear(15, 1))

    def forward(self, x):
        x = self.model(x)
        return x

import torch.nn as nn

class StartNet(nn.Module):

    def __init__(self):
        super(StartNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 20),
            nn.Dropout(),
            nn.Sigmoid(),
            nn.Linear(20, 1))

    def forward(self, x):
        x = self.model(x)
        return x

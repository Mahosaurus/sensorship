# pylint is disabled for this file because it is a PyTorch file and pylint does not understand PyTorch
# pylint: disable=E1101, no-name-in-module
from torch import tanh, nn

class LSTMModel(nn.Module):

    def __init__(self, seq_length=24):
        super(LSTMModel, self).__init__()

        self.num_classes = seq_length
        self.num_layers = 3
        self.input_size = 1
        self.hidden_size = 96

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)


        self.fc1 = nn.Linear(self.hidden_size, 100)
        self.fc2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(100, self.num_classes)

    def forward(self, x):
        # -> Shape of out = (Batch-Size, Target-Size, Hidden-Size)
        out, _ = self.lstm(x)#, (h0.detach(), c0.detach()))
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        # -> Shape of out = (Batch-Size, Target-Size)
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

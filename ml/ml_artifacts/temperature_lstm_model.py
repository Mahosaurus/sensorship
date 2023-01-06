import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self, seq_length=24):
        super(LSTMModel, self).__init__()
        
        self.num_classes = seq_length
        self.num_layers = 2
        self.input_size = 1
        self.hidden_size = 24
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # -> Shape of out = (Batch-Size, Target-Size,  Target-Size)
        out, hidden = self.lstm(x)#, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        # -> Shape of out = (Batch-Size, Target-Size)
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))

        return out

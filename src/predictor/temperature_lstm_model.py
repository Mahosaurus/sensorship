import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self, seq_length=24):
        super(LSTMModel, self).__init__()
        
        self.num_classes = seq_length
        self.num_layers = 1
        self.input_size = 1
        self.hidden_size = 20
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]
        #out = self.dropout(out)
        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
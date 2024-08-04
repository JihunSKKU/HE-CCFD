import torch
import torch.nn as nn
    
class ApproxSwish(nn.Module):
    def forward(self, x):
        return -0.002012*(x**4 - 73.2107355865 * x**2 - 248.508946322 * x - 59.5427435388)     

class CNN(nn.Module):
    def __init__(self, input_length:int=30, activation:str='ApproxSwish'):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=2)
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(64 * (input_length - 2), 64)
        self.drop4 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 1)

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Swish':
            self.activation = nn.SiLU()
        elif activation == 'ApproxSwish':
            self.activation = ApproxSwish()
        else:
            raise ValueError('Invalid activation function')

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.activation(x)
        x = self.drop2(x)
        
        x = x.view(x.size(0), -1)
        x = self.drop3(x)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop4(x)

        x = self.fc2(x)
        return torch.sigmoid(x)
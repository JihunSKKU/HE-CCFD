import torch
import torch.nn as nn

class Square(nn.Module):
    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x):
        return x ** 2
    
class ApproxReLU(nn.Module):
    def __init__(self):
        super(ApproxReLU, self).__init__()
        self.scale = 30

    def forward(self, x):
        x = x / self.scale
        
        # 4차
        # coeff = [0.0243987, 0.49096448, 1.08571579, 0.01212056, -0.69068458]
        
        # 8차        
        coeff = [0.0172036, 0.49211715, 1.90813097, 0.0811299, -5.34212661, -0.16520139, 7.32628553, 0.09354028, -3.44125495]
        
        result = 0
        for i in range(len(coeff)):
            result += coeff[i] * x ** i
        
        result *= self.scale
        return result

class CNN(nn.Module):
    def __init__(self, input_length, activation):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(64 * (input_length - 2), 64)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(64, 1)

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'ApproxReLU':
            self.activation = ApproxReLU()
        elif activation == 'Square':
            self.activation = Square()
        else:
            raise ValueError('Invalid activation function')

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.drop2(x)
        
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.drop3(x)

        x = self.fc2(x)
        return x

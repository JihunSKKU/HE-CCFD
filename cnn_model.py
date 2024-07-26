import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=0)
        self.drop1 = torch.nn.Dropout(0.1)
        self.batchnorm1 = torch.nn.BatchNorm1d(32)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.drop2 = torch.nn.Dropout(0.2)
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        self.drop3 = torch.nn.Dropout(0.4)
        self.fc1 = torch.nn.Linear(1792, 64)
        self.drop4 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.drop3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop4(x)
        x = self.fc2(x)
        return x
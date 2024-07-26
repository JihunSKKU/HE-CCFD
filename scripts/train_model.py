import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from src.models.cnn import CNN
from src.data.dataset import CreditCardDataset
from src.train.train import train

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    input_length = 30
    model = CNN(input_length).to(device)

    epochs = 30
    learning_rate = 0.0001

    train_dataset = CreditCardDataset(mode='train')
    test_dataset = CreditCardDataset(mode='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    best_model = train(model, train_loader, test_loader, epochs, learning_rate, device)
    torch.save(best_model.state_dict(), './models/best_model.pth')
    print(f'Model saved with best F1 Score')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from src.models.cnn import CNN
from src.data.dataset import CreditCardDataset
from src.utils.evaluate import evaluate

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    input_length = 30
    # activation <= 'ReLU', 'ApproxReLU', or 'Square'
    activation = 'ApproxReLU'
    model = CNN(input_length, activation).to(device)
    model.load_state_dict(torch.load(f'./models/best_{activation}_model.pth'))

    batch_size = 64
    test_dataset = CreditCardDataset(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"{activation} model evaluation")
    evaluate(model, test_loader, device)

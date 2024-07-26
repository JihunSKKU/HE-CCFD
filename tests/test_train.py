import unittest
import torch
from torch.utils.data import DataLoader
from src.models.cnn import CNN
from src.data.dataset import CreditCardDataset
from src.train.train import train

class TestTrain(unittest.TestCase):
    def test_training(self):
        input_length = 30
        model = CNN(input_length)
        device = torch.device('cpu')

        train_dataset = CreditCardDataset(mode='train')
        test_dataset = CreditCardDataset(mode='test')

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        train(model, train_loader, test_loader, epochs=1, learning_rate=0.0001, device=device)
        self.assertTrue(True)  # 단순히 훈련이 완료되는지 확인

if __name__ == '__main__':
    unittest.main()

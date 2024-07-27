import unittest
import torch
from src.models.cnn import CNN

class TestCNN(unittest.TestCase):
    def test_forward_pass(self):
        input_length = 30
        activation = 'ApproxReLU'
        model = CNN(input_length, activation=activation)
        x = torch.randn(1, 1, input_length)  # Batch size 1, 1 channel, input length
        output = model(x)
        self.assertEqual(output.shape, (1, 1))  # Batch size 1, 1 output

if __name__ == '__main__':
    unittest.main()

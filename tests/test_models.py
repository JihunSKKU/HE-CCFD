import unittest
import torch
from src.models.cnn import CNN
from torchsummary import summary

class TestCNN(unittest.TestCase):
    def test_forward_pass(self):
        input_length = 30
        # activation <= 'ReLU', 'ApproxReLU', or 'Square'
        activation = 'ApproxReLU'
        model = CNN(input_length, activation=activation).to('cpu')
        model.load_state_dict(torch.load(f'./models/best_{activation}_model.pth'))

        batch_size = 64
        x = torch.randn(batch_size, 1, input_length) 
        output = model(x)
        self.assertEqual(output.shape, (batch_size, 1))  # Batch size 1, 1 output

        summary(model, input_size=(1, input_length))

if __name__ == '__main__':
    unittest.main()

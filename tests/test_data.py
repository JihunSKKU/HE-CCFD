import unittest
from src.data.dataset import CreditCardDataset

class TestCreditCardDataset(unittest.TestCase):
    def test_dataset_length(self):
        dataset = CreditCardDataset(mode='train')
        self.assertGreater(len(dataset), 0)

    def test_get_item(self):
        dataset = CreditCardDataset(mode='train')
        x, y = dataset[0]
        self.assertEqual(x.shape[0], 1)  # 1D CNN input shape
        self.assertIsInstance(y.item(), int)

if __name__ == '__main__':
    unittest.main()

import torch
from torch.utils.data import Dataset
from .load_data import load_data_undersampling

class CreditCardDataset(Dataset):
    def __init__(self, mode='train'):
        if mode == 'train':
            self.x, _, _, self.y, _, _ = load_data_undersampling(log=True)
        elif mode == 'valid':
            _, self.x, _, _, self.y, _ = load_data_undersampling()
        elif mode == 'test':
            _, _, self.x, _, _, self.y = load_data_undersampling(log=True)
        else:
            raise ValueError('Argument of mode should be train, valid, or test.')

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if idx >= len(self.x):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.x)}")
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

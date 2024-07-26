import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data_undersampling(log: bool = False):
    if log:
        print("Data loading...")
    data = pd.read_csv("./datasets/creditcard.csv")

    # Undersampling
    fraud = data[data['Class'] == 1]
    normal = data[data['Class'] == 0]

    if log:
        print("Shape of fraud data:", fraud.shape)      # Number of fraud data is 492
        print("Shape of non-fraud data:", normal.shape) # Number of non-fraud data는 284315

    # We will use 492 fraud data and 4000 non-fraud data
    if log:
        print("\nUndersampling...")
    normal_undersampling = normal.sample(n=4000, random_state=42)

    data = pd.concat([fraud, normal_undersampling]).reset_index(drop=True)
    if log:
        print(data.Class.value_counts())

    # Splitting data
    X = data.drop('Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if log:
        print("\nData loading complete.\n")

    X_train_3d = np.expand_dims(X_train, axis=1)
    X_test_3d = np.expand_dims(X_test, axis=1)

    return X_train_3d, X_test_3d, y_train.reset_index(drop=True), y_test.reset_index(drop=True)

class CreditCardDataset(Dataset):
    def __init__(self, mode='train'):
        if mode == 'train':
            self.x, _, self.y, _ = load_data_undersampling()
        elif mode == 'test':
            _, self.x, _, self.y = load_data_undersampling()
        else:
            raise ValueError('Argument of mode should be train or test.')

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if idx >= len(self.x):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.x)}")
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)
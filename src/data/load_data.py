import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

import os

def load_data_undersampling(log: bool = False):
    X_train_path = "./data/X_train.csv"
    X_valid_path = "./data/X_valid.csv"
    X_test_path = "./data/X_test.csv"
    y_train_path = "./data/y_train.csv"
    y_valid_path = "./data/y_valid.csv"
    y_test_path = "./data/y_test.csv"

    if all(os.path.exists(path) for path in [X_train_path, X_valid_path, X_test_path, y_train_path, y_valid_path, y_test_path]):
        if log:
            print("Loading data from files...")

        X_train = pd.read_csv(X_train_path).values
        X_valid = pd.read_csv(X_valid_path).values
        X_test = pd.read_csv(X_test_path).values
        y_train = pd.read_csv(y_train_path).values.ravel()
        y_valid = pd.read_csv(y_valid_path).values.ravel()
        y_test = pd.read_csv(y_test_path).values.ravel()

        X_train_3d = np.expand_dims(X_train, axis=1)
        X_valid_3d = np.expand_dims(X_valid, axis=1)
        X_test_3d = np.expand_dims(X_test, axis=1)

        if log:
            print("\nData loading complete from files.\n")
    else:
        if log:
            print("Files not found. Loading and processing data...")

        data = pd.read_csv("./data/creditcard.csv")

        # Undersampling
        fraud = data[data['Class'] == 1]
        normal = data[data['Class'] == 0]

        if log:
            print("Shape of fraud data:", fraud.shape)
            print("Shape of non-fraud data:", normal.shape)

        normal_undersampling = normal.sample(n=4000, random_state=42)
        data = pd.concat([fraud, normal_undersampling]).reset_index(drop=True)
        if log:
            print(data.Class.value_counts())

        # Splitting data into initial train and temp sets
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Normalizing all features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        # X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        X_valid, X_test, y_valid, y_test = X_temp.copy(), X_temp.copy(), y_temp.copy(), y_temp.copy()

        # Save data to CSV files
        pd.DataFrame(X_train).to_csv(X_train_path, index=False)
        pd.DataFrame(X_valid).to_csv(X_valid_path, index=False)
        pd.DataFrame(X_test).to_csv(X_test_path, index=False)
        pd.DataFrame(y_train).to_csv(y_train_path, index=False)
        pd.DataFrame(y_valid).to_csv(y_valid_path, index=False)
        pd.DataFrame(y_test).to_csv(y_test_path, index=False)

        if log:
            print("\nData saved to files.\n")

        X_train_3d = np.expand_dims(X_train, axis=1)
        X_valid_3d = np.expand_dims(X_valid, axis=1)
        X_test_3d = np.expand_dims(X_test, axis=1)
        y_train = y_train.reset_index(drop=True)
        y_valid = y_valid.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

    return X_train_3d, X_valid_3d, X_test_3d, pd.Series(y_train), pd.Series(y_valid), pd.Series(y_test)


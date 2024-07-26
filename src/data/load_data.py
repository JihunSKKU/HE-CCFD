import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data_undersampling(log: bool = False):
    if log:
        print("Data loading...")
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

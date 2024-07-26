import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score

def train(model, train_loader, test_loader, epochs, learning_rate, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_f1 = 0
    best_model = None

    for epoch in range(epochs):
        print(f'Epoch {epoch+1} - ', end='')

        model.train()
        train_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        for x, label in train_loader:
            x, label = x.to(device), label.to(device).float()

            optimizer.zero_grad()
            output = model(x).squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(output) > 0.5).long()
            correct += (preds == label.long()).sum().item()
            total += label.size(0)
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        train_f1 = f1_score(all_labels, all_preds)
        print(f'loss: {train_loss / len(train_loader):.4f}, acc: {correct / total * 100:.2f}%, f1: {train_f1:.4f}', end=' / ')

        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for x, label in test_loader:
                x, label = x.to(device), label.to(device).float()

                output = model(x).squeeze()
                loss = criterion(output, label)

                test_loss += loss.item()
                preds = (torch.sigmoid(output) > 0.5).long()
                correct += (preds == label.long()).sum().item()
                total += label.size(0)
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        test_f1 = f1_score(all_labels, all_preds)
        print(f'test loss: {test_loss / len(test_loader):.4f}, test acc: {correct / total * 100:.2f}%, test f1: {test_f1:.4f}')

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model = model
    
    return best_model

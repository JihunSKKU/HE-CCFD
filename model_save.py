import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from load_data import CreditCardDataset
from cnn_model import CNN
from sklearn.metrics import f1_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, test_loader, epochs, learning_rate):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_f1 = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch+1} - ', end='')

        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        all_labels = []
        all_preds = []

        for batch_idx, batch in enumerate(train_loader):
            try:
                x, label = batch
            except IndexError as e:
                print(f"Skipping batch {batch_idx} due to IndexError: {e}")
                continue

            x, label = x.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(x).squeeze()
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(output) > 0.5).long()
            correct += (preds == label).sum().item()
            total += label.size(0)
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        train_f1 = f1_score(all_labels, all_preds)
        print(f'loss: {train_loss / len(train_loader):.4f}, acc: {correct / total * 100:.2f}%, f1: {train_f1:.4f}', end=' / ')

        # Test
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                try:
                    x, label = batch
                except IndexError as e:
                    print(f"Skipping batch {batch_idx} due to IndexError: {e}")
                    continue

                x, label = x.to(device), label.to(device)

                output = model(x).squeeze()
                loss = criterion(output, label.float())

                test_loss += loss.item()
                preds = (torch.sigmoid(output) > 0.5).long()
                correct += (preds == label).sum().item()
                total += label.size(0)
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        test_f1 = f1_score(all_labels, all_preds)
        print(f'test loss: {test_loss / len(test_loader):.4f}, test acc: {correct / total * 100:.2f}%, test f1: {test_f1:.4f}')

        # save model that has best F1 score
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), './models/best_model.pth')

if __name__ == '__main__':
    model = CNN().to(device)

    epochs = 30
    learning_rate = 0.0001

    train_dataset = CreditCardDataset(mode='train')
    test_dataset = CreditCardDataset(mode='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    train(model, train_loader, test_loader, epochs, learning_rate)

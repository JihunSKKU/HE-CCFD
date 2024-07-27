import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from copy import deepcopy

def train(model, train_loader, test_loader, epochs, learning_rate, device, pos_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_f1 = 0
    best_model = None

    train_losses = []
    train_f1_scores = []
    test_losses = []
    test_f1_scores = []

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')

        # Training phase
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

        train_losses.append(train_loss / len(train_loader))
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds)
        train_recall = recall_score(all_labels, all_preds)
        train_f1_scores.append(train_f1)

        print(f'Train - loss: {train_loss / len(train_loader):.4f}, acc: {train_acc * 100:.2f}%, precision: {train_precision:.4f}, recall: {train_recall:.4f}, f1: {train_f1:.4f}')

        # Testing phase
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

        test_losses.append(test_loss / len(test_loader))
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds)
        test_precision = precision_score(all_labels, all_preds)
        test_recall = recall_score(all_labels, all_preds)
        test_f1_scores.append(test_f1)

        print(f'Test  - loss: {test_loss / len(test_loader):.4f}, acc: {test_acc * 100:.2f}%, precision: {test_precision:.4f}, recall: {test_recall:.4f}, f1: {test_f1:.4f}')

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model = deepcopy(model)
    
    return best_model, train_losses, train_f1_scores, test_losses, test_f1_scores

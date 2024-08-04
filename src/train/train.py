import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from copy import deepcopy

def train(model, train_loader, valid_loader, epochs, learning_rate, device):
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_f1 = 0
    best_model = None

    train_losses = []
    train_f1_scores = []
    train_accuracies = []
    valid_losses = []
    valid_f1_scores = []
    valid_accuracies = []

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')

        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        train_losses.append(train_loss / len(train_loader))
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds)
        train_recall = recall_score(all_labels, all_preds)
        train_accuracies.append(train_acc)
        train_f1_scores.append(train_f1)

        print(f'Train - loss: {train_loss / len(train_loader):.4f}, acc: {train_acc * 100:.2f}%, precision: {train_precision:.4f}, recall: {train_recall:.4f}, f1: {train_f1:.4f}')

        # Validation phase
        model.eval()
        valid_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()

                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                preds = (outputs > 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        valid_losses.append(valid_loss / len(valid_loader))
        valid_acc = accuracy_score(all_labels, all_preds)
        valid_f1 = f1_score(all_labels, all_preds)
        valid_precision = precision_score(all_labels, all_preds)
        valid_recall = recall_score(all_labels, all_preds)
        valid_accuracies.append(valid_acc)
        valid_f1_scores.append(valid_f1)

        print(f'Valid - loss: {valid_loss / len(valid_loader):.4f}, acc: {valid_acc * 100:.2f}%, precision: {valid_precision:.4f}, recall: {valid_recall:.4f}, f1: {valid_f1:.4f}')

        if valid_f1 > best_f1:
            best_f1 = valid_f1
            best_model = deepcopy(model)
    
    return best_model, train_losses, train_f1_scores, train_accuracies, valid_losses, valid_f1_scores, valid_accuracies

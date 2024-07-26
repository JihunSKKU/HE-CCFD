import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def evaluate(model, test_loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for x, label in test_loader:
            x, label = x.to(device), label.to(device).float()
            output = model(x).squeeze()
            preds = (torch.sigmoid(output) > 0.5).long()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

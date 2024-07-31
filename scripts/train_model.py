import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from src.models.cnn import CNN
from src.data.dataset import CreditCardDataset
from src.train.train import train

import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    input_length = 30
    # activation <= 'ReLU', 'ApproxReLU', or 'Square'
    activation = 'ApproxReLU'
    model = CNN(input_length, activation).to(device)

    epochs = 100
    learning_rate = 0.00007
    batch_size = 64

    train_dataset = CreditCardDataset(mode='train')
    valid_dataset = CreditCardDataset(mode='valid')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    pos_weight = torch.tensor([2.0], dtype=torch.float).to(device)

    best_model, train_losses, train_f1_scores, train_accuracies, valid_losses, valid_f1_scores, valid_accuracies = train(
        model, train_loader, valid_loader, epochs, learning_rate, device, pos_weight)

    best_epoch = np.argmax(valid_f1_scores) + 1
    best_f1_score = valid_f1_scores[best_epoch - 1]
    torch.save(best_model.state_dict(), f'./models/best_{activation}_model.pth')
    print(f'\nModel saved with best F1 Score: {best_f1_score:.4f} at epoch {best_epoch}')

    epochs_range = range(1, epochs + 1)

    # Plot Loss graph
    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, valid_losses, label='Valid Loss')
    plt.scatter(best_epoch, valid_losses[best_epoch - 1], color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig(f'./images/{activation}_model_loss.png')
    print(f'Training loss plot saved as images/{activation}_model_loss.png')

    # Plot F1 Score graph
    plt.figure()
    plt.plot(epochs_range, train_f1_scores, label='Train F1 Score')
    plt.plot(epochs_range, valid_f1_scores, label='Valid F1 Score')
    plt.scatter(best_epoch, best_f1_score, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over Epochs')
    plt.legend()
    plt.savefig(f'./images/{activation}_model_f1.png')
    print(f'Training F1 Score plot saved as images/{activation}_model_f1.png')

    # Plot Accuracy graph
    plt.figure()
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, valid_accuracies, label='Valid Accuracy')
    plt.scatter(best_epoch, valid_accuracies[best_epoch - 1], color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig(f'./images/{activation}_model_accuracy.png')
    print(f'Training accuracy plot saved as images/{activation}_model_accuracy.png')

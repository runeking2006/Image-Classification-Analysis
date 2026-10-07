import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

def evaluate_model(model, test_loader, classes, device='cpu'):
    """
    Evaluates the model on test data and returns accuracy and predictions.
    """
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = 100. * correct / total
    return acc, all_preds, all_labels

def plot_confusion_matrix(labels, preds, classes, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(history_cnn, history_densenet, save_path_prefix):
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history_cnn['train_loss'], label='CNN Train', color='blue', linestyle='-')
    plt.plot(history_cnn['val_loss'], label='CNN Val', color='blue', linestyle='--')
    plt.plot(history_densenet['train_loss'], label='DenseNet Train', color='red', linestyle='-')
    plt.plot(history_densenet['val_loss'], label='DenseNet Val', color='red', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path_prefix}_loss.png")
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history_cnn['train_acc'], label='CNN Train', color='blue', linestyle='-')
    plt.plot(history_cnn['val_acc'], label='CNN Val', color='blue', linestyle='--')
    plt.plot(history_densenet['train_acc'], label='DenseNet Train', color='red', linestyle='-')
    plt.plot(history_densenet['val_acc'], label='DenseNet Val', color='red', linestyle='--')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path_prefix}_accuracy.png")
    plt.close()

import torch
import random
import numpy as np
from dataset import get_dataloaders
from models import SimpleCNN, MiniDenseNet
from train import train_model
from evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def visualize_features(model, img_tensor, save_path, device='cpu'):
    model.eval()
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        _, features = model(img_tensor, return_features=True)
        
    features = features[0].cpu() # Get first image in batch
    # Plot first 16 feature maps
    num_maps = min(16, features.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(num_maps):
        ax = axes[i//4, i%4]
        ax.imshow(features[i].numpy(), cmap='viridis')
        ax.axis('off')
    plt.suptitle("Feature Maps (Layer output)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading CIFAR-10...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=64, root_dir='../dataset')
    
    # 2. Initialize Models
    print("Initializing models...")
    cnn = SimpleCNN(num_classes=10)
    densenet = MiniDenseNet(num_classes=10, growth_rate=12, block_layers=[3, 3, 3])
    
    # Analyze parameters
    cnn_total, cnn_train = count_parameters(cnn)
    dense_total, dense_train = count_parameters(densenet)
    
    print(f"CNN Parameters: {cnn_total} (Trainable: {cnn_train})")
    print(f"DenseNet Parameters: {dense_total} (Trainable: {dense_train})")
    
    # 3. Training
    epochs = 5
    print(f"Training Conventional CNN for {epochs} epochs...")
    cnn_history = train_model(cnn, train_loader, val_loader, epochs=epochs, lr=0.001, device=device)
    
    print(f"Training DenseNet for {epochs} epochs...")
    dense_history = train_model(densenet, train_loader, val_loader, epochs=epochs, lr=0.001, device=device)
    
    # 4. Evaluation
    print("Evaluating CNN on Test Set...")
    cnn_acc, cnn_preds, cnn_labels = evaluate_model(cnn, test_loader, classes, device=device)
    
    print("Evaluating DenseNet on Test Set...")
    dense_acc, dense_preds, dense_labels = evaluate_model(densenet, test_loader, classes, device=device)
    
    print(f"Final Test Accuracy - CNN: {cnn_acc:.2f}%, DenseNet: {dense_acc:.2f}%")
    
    # 5. Results Generation
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)
    
    # Plot training curves
    plot_training_curves(cnn_history, dense_history, str(results_dir / 'training_curves'))
    
    # Plot confusion matrices
    plot_confusion_matrix(cnn_labels, cnn_preds, classes, str(results_dir / 'confusion_matrix_cnn.png'))
    plot_confusion_matrix(dense_labels, dense_preds, classes, str(results_dir / 'confusion_matrix_densenet.png'))
    
    # Parameter comparison chart
    plt.figure(figsize=(6, 4))
    plt.bar(['CNN', 'DenseNet'], [cnn_total, dense_total], color=['blue', 'red'])
    plt.title('Total Parameters Comparison')
    plt.ylabel('Number of Parameters')
    for i, v in enumerate([cnn_total, dense_total]):
        plt.text(i, v + 1000, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(str(results_dir / 'parameter_comparison.png'))
    plt.close()
    
    # Final Accuracy Bar Chart
    plt.figure(figsize=(6, 4))
    plt.bar(['CNN', 'DenseNet'], [cnn_acc, dense_acc], color=['blue', 'red'])
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    for i, v in enumerate([cnn_acc, dense_acc]):
        plt.text(i, v + 2, f"{v:.2f}%", ha='center')
    plt.tight_layout()
    plt.savefig(str(results_dir / 'accuracy_comparison.png'))
    plt.close()
    
    # Save CSV
    df = pd.DataFrame({
        'Model': ['CNN', 'DenseNet'],
        'Total_Parameters': [cnn_total, dense_total],
        'Trainable_Parameters': [cnn_train, dense_train],
        'Test_Accuracy': [cnn_acc, dense_acc],
        'Training_Time_s': [cnn_history['time'], dense_history['time']]
    })
    df.to_csv(str(results_dir / 'model_comparison.csv'), index=False)
    
    # Visualize features for first test image
    sample_img, _ = next(iter(test_loader))
    sample_img = sample_img[:1] # 1 image
    visualize_features(cnn, sample_img, str(results_dir / 'cnn_features.png'), device)
    visualize_features(densenet, sample_img, str(results_dir / 'densenet_features.png'), device)
    
    print("Project run completed. Results saved in 'results/' directory.")

if __name__ == "__main__":
    main()

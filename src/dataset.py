import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path

def get_dataloaders(batch_size=64, root_dir='../dataset'):
    """
    Downloads and prepares CIFAR-10 data loaders.
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    
    # Common transformations: normalize and convert to tensor
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download training dataset
    train_dataset_full = datasets.CIFAR10(
        root=str(root), train=True, download=True, transform=transform_train
    )
    
    # Split train into train and validation (45k train, 5k val)
    train_size = 45000
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset_full, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Re-apply test transform to validation dataset for fair evaluation
    val_dataset.dataset.transform = transform_test

    # Download test dataset
    test_dataset = datasets.CIFAR10(
        root=str(root), train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
               
    return train_loader, val_loader, test_loader, classes

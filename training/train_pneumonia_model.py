"""
Training Script for Pneumonia Classification Model

This script trains a DenseNet121-based model to classify chest X-rays as pneumonia or normal.

Dataset Requirements:
- Directory structure:
  data/
    pneumonia_dataset/
      train/
        pneumonia/
        normal/
      val/
        pneumonia/
        normal/
      test/
        pneumonia/
        normal/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

# Configuration
CONFIG = {
    'data_dir': './data/pneumonia_dataset',
    'model_save_path': './agents/image_analysis_agent/pneumonia_agent/models/pneumonia_classification_model.pth',
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'image_size': 224,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class PneumoniaClassifier(nn.Module):
    """Pneumonia classification model based on DenseNet121."""
    
    def __init__(self, num_classes=2):
        super(PneumoniaClassifier, self).__init__()
        # Load pre-trained DenseNet121
        self.backbone = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
        num_ftrs = self.backbone.classifier.in_features
        # Replace classifier for binary classification
        self.backbone.classifier = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def get_data_loaders(data_dir, batch_size, image_size, num_workers):
    """Create data loaders for training, validation, and testing."""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation/test
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """Train the pneumonia classification model."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*train_correct/train_total:.2f}%'})
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*val_correct/val_total:.2f}%'})
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f'  ✓ Saved best model (Val Acc: {best_val_acc:.2f}%)')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the model on test set."""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./training/pneumonia_confusion_matrix.png')
    print("\nConfusion matrix saved to ./training/pneumonia_confusion_matrix.png")
    
    return all_preds, all_labels

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accuracies, label='Train Acc')
    ax2.plot(val_accuracies, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('./training/pneumonia_training_history.png')
    print("Training history saved to ./training/pneumonia_training_history.png")

def main():
    """Main training function."""
    
    print("="*60)
    print("Pneumonia Classification Model Training")
    print("="*60)
    
    # Check if data directory exists
    if not os.path.exists(CONFIG['data_dir']):
        print(f"ERROR: Data directory not found: {CONFIG['data_dir']}")
        print("\nPlease prepare your dataset with the following structure:")
        print("data/pneumonia_dataset/")
        print("  train/")
        print("    pneumonia/")
        print("    normal/")
        print("  val/")
        print("    pneumonia/")
        print("    normal/")
        print("  test/")
        print("    pneumonia/")
        print("    normal/")
        return
    
    # Create training directory
    os.makedirs('./training', exist_ok=True)
    
    # Get data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        CONFIG['data_dir'],
        CONFIG['batch_size'],
        CONFIG['image_size'],
        CONFIG['num_workers']
    )
    
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = PneumoniaClassifier(num_classes=len(class_names))
    model = model.to(CONFIG['device'])
    print(f"Model initialized on {CONFIG['device']}")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader,
        CONFIG['num_epochs'], CONFIG['learning_rate'], CONFIG['device']
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(CONFIG['model_save_path']))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluate_model(model, test_loader, CONFIG['device'], class_names)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Model saved to: {CONFIG['model_save_path']}")
    print("="*60)

if __name__ == '__main__':
    main()


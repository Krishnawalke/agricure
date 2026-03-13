"""
AgriCure - Plant Disease Detection Model
Uses Transfer Learning with MobileNetV2 (lightweight, fast, accurate)
Dataset: PlantVillage (38 disease classes)

Install requirements first:
  pip install torch torchvision pillow numpy scikit-learn matplotlib

Download dataset from:
  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
  Extract to: ./dataset/plantvillage/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import os
import json
import time

# ═══════════════════════════════
# CONFIG
# ═══════════════════════════════
CONFIG = {
    'data_dir': './dataset/plantvillage/color',
    'model_save_path': './model/plant_disease_model.pth',
    'classes_save_path': './model/classes.json',
    'image_size': 224,
    'batch_size': 64,
    'num_epochs': 15,
    'learning_rate': 0.0001,
    'val_split': 0.2,
    'num_workers': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"🌿 AgriCure ML Model Trainer")
print(f"📱 Using device: {CONFIG['device']}")

# ═══════════════════════════════
# DATA TRANSFORMS
# ═══════════════════════════════
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ═══════════════════════════════
# LOAD DATASET
# ═══════════════════════════════
def load_data():
    print(f"\n📂 Loading dataset from: {CONFIG['data_dir']}")
    
    if not os.path.exists(CONFIG['data_dir']):
        print(f"❌ Dataset not found at {CONFIG['data_dir']}")
        print("Please download PlantVillage dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        return None, None, None

    full_dataset = datasets.ImageFolder(CONFIG['data_dir'], transform=train_transforms)
    
    val_size = int(len(full_dataset) * CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply proper transforms to val set
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=CONFIG['num_workers'])

    classes = full_dataset.classes
    print(f"✅ Dataset loaded: {len(full_dataset)} images, {len(classes)} classes")
    print(f"   Train: {train_size} | Val: {val_size}")
    
    return train_loader, val_loader, classes

# ═══════════════════════════════
# BUILD MODEL
# ═══════════════════════════════
def build_model(num_classes):
    print(f"\n🔧 Building MobileNetV2 model for {num_classes} classes...")
    
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)
    
    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Unfreeze last 3 layers for fine-tuning
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    
    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    print("✅ Model built successfully")
    return model

# ═══════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════
def train_model(model, train_loader, val_loader, num_classes):
    device = CONFIG['device']
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n🚀 Starting training for {CONFIG['num_epochs']} epochs...")
    
    for epoch in range(CONFIG['num_epochs']):
        start_time = time.time()
        
        # ── TRAIN ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r')
        
        # ── VALIDATE ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc   = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('./model', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes
            }, CONFIG['model_save_path'])
            print(f"  ✅ Best model saved! Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
    
    print(f"\n🎉 Training complete! Best Val Accuracy: {best_val_acc:.2f}%")
    return history

# ═══════════════════════════════
# MAIN
# ═══════════════════════════════
if __name__ == '__main__':
    train_loader, val_loader, classes = load_data()
    
    if train_loader is None:
        exit(1)
    
    # Save class names
    os.makedirs('./model', exist_ok=True)
    with open(CONFIG['classes_save_path'], 'w') as f:
        json.dump(classes, f, indent=2)
    print(f"✅ Classes saved to {CONFIG['classes_save_path']}")
    
    # Build and train
    model = build_model(len(classes))
    history = train_model(model, train_loader, val_loader, len(classes))
    
    print("\n📊 Training Summary:")
    print(f"  Final Train Acc: {history['train_acc'][-1]:.2f}%")
    print(f"  Final Val Acc:   {history['val_acc'][-1]:.2f}%")
    print(f"\n✅ Model saved to: {CONFIG['model_save_path']}")
    print("✅ Run 'python server.py' to start the API server")

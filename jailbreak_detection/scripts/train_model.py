import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.model import JailbreakDetector

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class JailbreakDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        
        # Load clean images
        clean_dir = os.path.join(data_dir, 'clean')
        if os.path.exists(clean_dir):
            for img_file in os.listdir(clean_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(clean_dir, img_file))
                    self.labels.append(0)
        
        # Load adversarial images
        adv_dir = os.path.join(data_dir, 'adversarial')
        if os.path.exists(adv_dir):
            for img_file in os.listdir(adv_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(adv_dir, img_file))
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')
        image = image.resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(PROJECT_ROOT, 'data', 'clean'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'data', 'adversarial'), exist_ok=True)
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)
    
    # Initialize model
    model = JailbreakDetector().to(device)
    
    # Create dataset and dataloader
    dataset = JailbreakDataset(os.path.join(PROJECT_ROOT, 'data'))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
        
        # Save best model
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model with accuracy: {best_accuracy:.2f}%")
            print(f"Model saved at: {model_path}")
    
    print("\nTraining completed!")
    print(f"Best accuracy achieved: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main() 
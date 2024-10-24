import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
import torch.nn.functional as F
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# Define device


class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body','Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']# Define each class columns 


class FocalLoss(nn.Module):# Focal Loss function
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * ((1 - p_t) ** self.gamma)
        log_p_t = torch.log(p_t + 1e-8)
        loss = -focal_weight * log_p_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Model for classification
class attention_model(nn.Module):
    def __init__(self, in_channels, attention_size):
        super(attention_model, self).__init__()
        self.query = nn.Conv2d(in_channels, attention_size, kernel_size=1)
        self.key = nn.Conv2d(in_channels, attention_size, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, c, h, w = x.size()
        query = self.query(x).view(batch_size, -1, h * w) # Create query, key, and value tensors
        key = self.key(x).view(batch_size, -1, h * w)  
        value = self.value(x).view(batch_size, c, h * w) 
        attention_scores = torch.bmm(query.transpose(1, 2), key)  # Compute attention scores
        attention_scores = attention_scores / (h * w) ** 0.5 
        attention_map = self.softmax(attention_scores)# Apply softmax to get attention map
        out = torch.bmm(value, attention_map.transpose(1, 2))  # Weighted sum of value tensor 
        out = out.view(batch_size, c, h, w) 
        out = out + x# Add residual connection
        return out
      
class model_mobilenetv3(nn.Module):
    def __init__(self, num_classes=10, attention_size=64):
        super(model_mobilenetv3, self).__init__(
        self.mobilenet_v3 = models.mobilenet_v3_large(weights='DEFAULT') # Load the pre-trained MobileNetV3 Large model
        in_features = self.mobilenet_v3.classifier[0].in_features# Get the number of input features for the classifier layer
        self.self_attention = attention_model(in_channels=in_features, attention_size=attention_size)# Add self-attention block after certain layer
        self.mobilenet_v3.classifier = nn.Sequential(  # Replace the classifier layer to match the number of classes
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.mobilenet_v3.features(x) # Pass through MobileNetV3 backbone
        x = self.self_attention(x)# Apply self-attention block
        x = F.adaptive_avg_pool2d(x, (1, 1)) # Global Average Pooling (same as original MobileNetV3)
        x = torch.flatten(x, 1)
        x = self.mobilenet_v3.classifier(x)# Classification layer
        return x


train_transform = transforms.Compose([     # Define transformations for training
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Function to calculate class weights
def calculate_class_weights(dataset):
    labels = [label for _, label in dataset] # Extract labels from the dataset
    unique_classes = np.unique(labels)# Get unique classes present in the dataset
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)# Compute class weights only for the unique classes present
    return torch.tensor(class_weights, dtype=torch.float32)# Convert class weights to tensor

# Function to create data loaders
def create_train_dataloader(train_dir, batch_size=32):
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    print(f"Number of images in train dataset: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, train_dataset

# Training loop
def train_model(train_loader, model, criterion, optimizer, num_epochs=50, output_folder='training_results'):
    os.makedirs(output_folder, exist_ok=True)
    model.to(device)
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

    
    model_path = os.path.join(output_folder, 'trained_model.pth')# Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save training losses and accuracies
    torch.save({
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }, os.path.join(output_folder, 'training_metrics.pth'))


if __name__ == "__main__":
    train_dir = 'training'
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50

    train_loader, train_dataset = create_train_dataloader(train_dir, batch_size=batch_size)
    model = model_mobilenetv3(num_classes=10)
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_model(train_loader, model, criterion, optimizer, num_epochs=num_epochs)

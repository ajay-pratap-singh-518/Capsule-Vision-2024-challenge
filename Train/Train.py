import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, precision_recall_curve, auc, recall_score, f1_score,balanced_accuracy_score
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# Define device 

class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']# Define class columns 

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs and targets have the same shape
        if inputs.size(0) != targets.size(0):
            raise ValueError("Shape mismatch between inputs and targets")
        
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = self.alpha * ((1 - p_t) ** self.gamma)
        
        log_p_t = torch.log(p_t + 1e-8)
        loss = -focal_weight * log_p_t
        
        # Reduce loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



def create_dataloaders(train_dir, val_dir, batch_size=32):# Define data loaders for training and validation
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)
    print(f"Number of images in train dataset: {len(train_dataset)}")
    print(f"Number of images in validation dataset: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset, val_dataset
    
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
        attention_map = self.softmax(attention_scores) # Apply softmax to get attention map
        out = torch.bmm(value, attention_map.transpose(1, 2))  # (batch_size, in_channels, h * w)# Weighted sum of value tensor
        out = out.view(batch_size, c, h, w)  # Reshape back to original dimensions
        out = out + x # Add residual connection
        return out
class model_mobilenetv3(nn.Module):
    def __init__(self, num_classes=10, attention_size=64):
        super(model_mobilenetv3, self).__init__()
        self.mobilenet_v3 = models.mobilenet_v3_large(weights='DEFAULT')# Load the pre-trained MobileNetV3 Large model
        in_features = self.mobilenet_v3.classifier[0].in_features # Get the number of input features for the classifier layer
        self.self_attention = attention_model(in_channels=in_features, attention_size=attention_size)# Add self-attention block after certain layers
        self.mobilenet_v3.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        x = self.mobilenet_v3.features(x) # Pass through MobileNetV3 backbone
        x = self.self_attention(x)# Apply self-attention block
        x = F.adaptive_avg_pool2d(x, (1, 1)) # Global Average Pooling (same as original MobileNetV3)
        x = torch.flatten(x, 1)
        x = self.mobilenet_v3.classifier(x) # Classification layer
        return x


# Define the transformations for training and validation separately
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_class_weights(dataset):# Function to calculate class weights
    labels = [label for _, label in dataset]
    class_weights = compute_class_weight('balanced', classes=np.arange(10), y=labels)
    return torch.tensor(class_weights, dtype=torch.float32)

def save_predictions_to_excel(predictions, labels, logits, class_names, output_path):
    if not (len(predictions) == len(labels) == len(logits)):
        print("Mismatch detected:")
        print(f"Length of predictions: {len(predictions)}")
        print(f"Length of labels: {len(labels)}")
        print(f"Length of logits: {len(logits)}")
        raise ValueError("Mismatched lengths: Ensure that predictions, labels, and logits all have the same length.")
    logits_tensor = torch.tensor(logits)# Convert logits to a tensor
    if logits_tensor.ndimension() == 1:
        logits_tensor = logits_tensor.unsqueeze(0)  # Adding batch dimension if missing
    probs = torch.softmax(logits_tensor, dim=1).numpy()# Apply softmax to logits to get probabilities
    if probs.ndim != 2 or probs.shape[1] != len(class_names): # Check the shape of probs
        print(f"Shape of probabilities: {probs.shape}")
        raise ValueError("Unexpected shape for probabilities: Ensure that the number of classes matches the second dimension of probs.")
    df = pd.DataFrame({
        "True Labels": [class_names[label] for label in labels],
        "Predicted Labels": [class_names[pred] for pred in predictions]
    })# Create a DataFrame to store the data
    for i, class_name in enumerate(class_names):
        if i < probs.shape[1]:
            df[f'Probability {class_name}'] = probs[:, i]
        else:
            print(f"Index {i} is out of bounds for probabilities with shape {probs.shape}")
    df.to_excel(output_path, index=False)# Save the DataFrame to an Excel file
    print(f"Predictions and probabilities saved to {output_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, output_path=None):# Function to plot confusion matrix
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: Empty true or predicted labels. Skipping confusion matrix plotting.")
        return 
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()

    plt.close()

def plot_roc_curve(labels, preds, num_classes, output_path=None):# Function to plot ROC curve
    preds_tensor = torch.tensor(preds)# Convert preds to a tensor and print its shape
    print(f"Predictions tensor shape: {preds_tensor.shape}")
    if preds_tensor.dim() != 2:
        raise ValueError("Predictions tensor should be 2D with shape (batch_size, num_classes).")
    labels_one_hot = np.eye(num_classes)[labels]# Convert labels to one-hot encoding
    preds_prob = torch.softmax(preds_tensor, dim=1).numpy()# Apply softmax to get class probabilities
    print(f"Predictions probabilities shape: {preds_prob.shape}")
    fpr, tpr, _ = roc_curve(labels_one_hot.ravel(), preds_prob.ravel())# Compute ROC curve and AUC
    auc_roc = roc_auc_score(labels_one_hot, preds_prob, multi_class='ovr')
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC-ROC Curve (AUC = {auc_roc:.2f})')
    plt.legend(loc="lower right")
    if output_path:
        plt.savefig(output_path)
        print(f"ROC curve saved to {output_path}")
    else:
        plt.show()
    plt.close()
def plot_roc_curves_by_class(labels, preds, num_classes, class_names, output_path=None):
    preds_tensor = torch.tensor(preds) # Convert predictions to tensor
    if preds_tensor.dim() != 2:
        raise ValueError("Predictions tensor should be 2D with shape (batch_size, num_classes).")
    labels_one_hot = np.eye(num_classes)[labels]# Convert labels to one-hot encoding
    preds_prob = torch.softmax(preds_tensor, dim=1).numpy()# Apply softmax to get class probabilities
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_one_hot[:, i], preds_prob[:, i])
        auc_roc = roc_auc_score(labels_one_hot[:, i], preds_prob[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (area = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc='lower right')
    if output_path:
        plt.savefig(output_path)
        print(f"ROC curves for each class saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
def calculate_specificity(y_true, y_pred):# Function to calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def generate_metrics_report(y_true, y_pred, class_columns, output_path=None):
    metrics_report = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) == 1:  # Assuming y_true is already a 1D array of class indices
        y_true_classes = y_true
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class predictions if needed
    else:  # Assuming y_true is a 2D array of one-hot encoded labels
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)

    unique_true_classes = len(set(y_true_classes))
    num_classes = len(class_columns)
    print(f"Number of classes in dataset: {num_classes}")

    print(f"Number of unique true classes: {unique_true_classes}")
    print("Unique labels in y_true:", np.unique(y_true_classes))
    print("Unique predictions:", np.unique(y_pred_classes))
    print("Class columns:", class_columns)

    if len(class_columns) != unique_true_classes:
        print(f"Warning: Mismatch in class names and number of classes")
        class_columns = [f'Class_{i}' for i in range(unique_true_classes)]

    class_report = classification_report(y_true_classes, y_pred_classes, target_names=class_columns[:unique_true_classes], output_dict=True, zero_division=0)

    auc_roc_scores = {}
    for i, class_name in enumerate(class_columns[:unique_true_classes]):
        if len(y_true.shape) == 2:  # If y_true is one-hot encoded
            auc_roc_scores[class_name] = roc_auc_score(y_true[:, i], y_pred[:, i])
        else:
            auc_roc_scores[class_name] = roc_auc_score(y_true == i, y_pred[:, i])

    mean_auc_roc = np.mean(list(auc_roc_scores.values()))
    auc_roc_scores['mean_auc'] = mean_auc_roc

    specificity_scores = {}
    for i, class_name in enumerate(class_columns[:unique_true_classes]):
        if len(y_true.shape) == 2:  # If y_true is one-hot encoded
            specificity_scores[class_name] = calculate_specificity(y_true[:, i], (y_pred[:, i] > 0.5).astype(int))
        else:
            specificity_scores[class_name] = calculate_specificity(y_true == i, (y_pred[:, i] > 0.5).astype(int))

    mean_specificity = np.mean(list(specificity_scores.values()))
    specificity_scores['mean_specificity'] = mean_specificity

    average_precision_scores = {}
    for i, class_name in enumerate(class_columns[:unique_true_classes]):
        if len(y_true.shape) == 2:  # If y_true is one-hot encoded
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        else:
            precision, recall, _ = precision_recall_curve(y_true == i, y_pred[:, i])
        average_precision_scores[class_name] = auc(recall, precision)

    mean_average_precision = np.mean(list(average_precision_scores.values()))
    average_precision_scores['mean_average_precision'] = mean_average_precision

    sensitivity_scores = {}
    for i, class_name in enumerate(class_columns[:unique_true_classes]):
        if len(y_true.shape) == 2:  # If y_true is one-hot encoded
            sensitivity_scores[class_name] = recall_score(y_true[:, i], (y_pred[:, i] > 0.5).astype(int), average='binary', zero_division=0)
        else:
            sensitivity_scores[class_name] = recall_score(y_true == i, (y_pred[:, i] > 0.5).astype(int), average='binary', zero_division=0)

    mean_sensitivity = np.mean(list(sensitivity_scores.values()))
    sensitivity_scores['mean_sensitivity'] = mean_sensitivity

    f1_scores = {}
    for i, class_name in enumerate(class_columns[:unique_true_classes]):
        if len(y_true.shape) == 2:  # If y_true is one-hot encoded
            f1_scores[class_name] = f1_score(y_true[:, i], (y_pred[:, i] > 0.5).astype(int), average='binary', zero_division=0)
        else:
            f1_scores[class_name] = f1_score(y_true == i, (y_pred[:, i] > 0.5).astype(int), average='binary', zero_division=0)

    mean_f1_score = np.mean(list(f1_scores.values()))
    f1_scores['mean_f1_score'] = mean_f1_score
    balanced_accuracy_scores = balanced_accuracy_score(y_true_classes, y_pred_classes) if y_true.shape[0] > 0 else 0.0
    metrics_report['classification_report'] = class_report
    metrics_report['roc_auc_scores'] = auc_roc_scores
    metrics_report['specificity_scores'] = specificity_scores
    metrics_report['average_precision_scores'] = average_precision_scores
    metrics_report['sensitivity_scores'] = sensitivity_scores
    metrics_report['f1_scores'] = f1_scores
    metrics_report['balanced_accuracy'] = balanced_accuracy_scores
    metrics_report_json = json.dumps(metrics_report, indent=4)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(metrics_report_json)
        print(f"Metrics Report saved to: {output_path}")

    print("Metrics Report:")
    print(metrics_report_json)

    return metrics_report, metrics_report_json


def save_epoch_metrics_to_excel(epoch_metrics, output_path):# Function to save epoch metrics to an Excel sheet
    df = pd.DataFrame(epoch_metrics)
    df.to_excel(output_path, index=False)
    print(f"Epoch metrics saved to {output_path}")



def train_and_validate_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=50, output_folder='model_mobilenet_v3'):# Training and validation loop
    os.makedirs(output_folder, exist_ok=True)
    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epoch_metrics = []
    for epoch in range(num_epochs):# Training Phase
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
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

        # Validation Phase
        model.eval()
        val_labels = []
        val_preds = []
        val_probs = []
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)
                
                # Calculate accuracy
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                running_val_loss += loss.item() * images.size(0)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')
    # Append metrics for this epoch
        epoch_metrics.append({
            'Epoch': epoch + 1,
            'Training Loss': epoch_loss,
            'Training Accuracy (%)': train_accuracy,
            'Validation Loss': val_loss,
            'Validation Accuracy (%)': val_accuracy
        })
    # Save predictions and metrics
    save_predictions_to_excel(
        predictions=val_preds,
        labels=val_labels,
        logits=val_probs,
        class_names=class_columns,
        output_path=os.path.join(output_folder, 'predictions.xlsx')
    )

    plot_confusion_matrix(
        y_true=val_labels,
        y_pred=val_preds,
        class_names=class_columns,
        normalize=True,
        output_path=os.path.join(output_folder, 'confusion_matrix.png')
    )

    plot_roc_curve(
        labels=val_labels,
        preds=val_probs,
        num_classes=len(class_columns),
        output_path=os.path.join(output_folder, 'roc_curve.png')
    )
    plot_roc_curves_by_class(
        labels=val_labels,
        preds=val_probs,
        num_classes=len(class_columns),
        class_names=class_columns,
        output_path=os.path.join(output_folder, 'roc_curves_by_class.png')
    )
    
    generate_metrics_report(
        y_true=val_labels,
        y_pred=val_probs,
        class_columns=class_columns,
        output_path=os.path.join(output_folder, 'metrics_report.json')
    )

    # Save the trained model
    model_path = os.path.join(output_folder, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot accuracy and loss graphs
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'accuracy_graph.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'loss_graph.png'))
    plt.close()
    # Save epoch metrics to Excel
    save_epoch_metrics_to_excel(
        epoch_metrics,
        output_path=os.path.join(output_folder, 'epoch_metrics.xlsx')
    )
batch_size = 32
learning_rate = 1e-4
num_epochs = 50
train_dir = 'training'
val_dir = 'validation'
train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(train_dir, val_dir, batch_size=batch_size)

model = model_mobilenetv3(num_classes=len(class_columns))
class_weights = calculate_class_weights(train_dataset).to(device)
criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_and_validate_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=50)

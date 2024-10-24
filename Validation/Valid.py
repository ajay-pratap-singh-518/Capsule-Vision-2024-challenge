import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, precision_recall_curve, auc, recall_score, f1_score, balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import torchvision.models as models
import torch.nn.functional as F
# Define device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define class columns globally
class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',
                 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']

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



class SelfAttention(nn.Module):
    def __init__(self, in_channels, attention_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, attention_size, kernel_size=1)
        self.key = nn.Conv2d(in_channels, attention_size, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, c, h, w = x.size()
        
        # Create query, key, and value tensors
        query = self.query(x).view(batch_size, -1, h * w)  # (batch_size, attention_size, h * w)
        key = self.key(x).view(batch_size, -1, h * w)  # (batch_size, attention_size, h * w)
        value = self.value(x).view(batch_size, c, h * w)  # (batch_size, in_channels, h * w)
        
        # Compute attention scores
        attention_scores = torch.bmm(query.transpose(1, 2), key)  # (batch_size, h * w, h * w)
        attention_scores = attention_scores / (h * w) ** 0.5  # Scaled dot-product attention
        
        # Apply softmax to get attention map
        attention_map = self.softmax(attention_scores)  # (batch_size, h * w, h * w)
        
        # Weighted sum of value tensor
        out = torch.bmm(value, attention_map.transpose(1, 2))  # (batch_size, in_channels, h * w)
        out = out.view(batch_size, c, h, w)  # Reshape back to original dimensions
        
        # Add residual connection
        out = out + x
        
        return out
class MobileNetV3ClassifierWithAttention(nn.Module):
    def __init__(self, num_classes=10, attention_size=64):
        super(MobileNetV3ClassifierWithAttention, self).__init__()
        # Load the pre-trained MobileNetV3 Large model
        self.mobilenet_v3 = models.mobilenet_v3_large(weights='DEFAULT')
        
        # Get the number of input features for the classifier layer
        in_features = self.mobilenet_v3.classifier[0].in_features
        
        # Add self-attention block after certain layers
        self.self_attention = SelfAttention(in_channels=in_features, attention_size=attention_size)
        
        # Replace the classifier layer to match the number of classes
        self.mobilenet_v3.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # Pass through MobileNetV3 backbone
        x = self.mobilenet_v3.features(x)
        
        # Apply self-attention block
        x = self.self_attention(x)
        
        # Global Average Pooling (same as original MobileNetV3)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Classification layer
        x = self.mobilenet_v3.classifier(x)
        
        return x



# Define transformations for validation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Function to create validation data loader
def create_val_dataloader(val_dir, batch_size=32):
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)
    print(f"Number of images in validation dataset: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return val_loader, val_dataset

# Validation function
def validate_model(val_loader, model, criterion, output_folder='atten_focal_equal/validation_results_mobv3'):
    os.makedirs(output_folder, exist_ok=True)
    model.to(device)
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
    print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')
    
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

# Helper functions (same as in your original code)
def save_predictions_to_excel(predictions, labels, logits, class_names, output_path):
    # Ensure all inputs have the same length
    if not (len(predictions) == len(labels) == len(logits)):
        print("Mismatch detected:")
        print(f"Length of predictions: {len(predictions)}")
        print(f"Length of labels: {len(labels)}")
        print(f"Length of logits: {len(logits)}")
        raise ValueError("Mismatched lengths: Ensure that predictions, labels, and logits all have the same length.")

    probs = np.array(logits)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        "True Labels": [class_names[label] for label in labels],
        "Predicted Labels": [class_names[pred] for pred in predictions]
    })
    for i, class_name in enumerate(class_names):
        df[f'Probability {class_name}'] = probs[:, i]

    # Save the DataFrame to an Excel file
    df.to_excel(output_path, index=False)
    print(f"Predictions and probabilities saved to {output_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, output_path=None, fontsize=14, tick_fontsize=10):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(18,16))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": fontsize})  # Increase annotation font size
    
    plt.xlabel('Predicted Labels', fontsize=fontsize + 2)  # Increase font size of x-label
    plt.ylabel('True Labels', fontsize=fontsize + 2)  # Increase font size of y-label
    plt.title('Confusion Matrix', fontsize=fontsize + 4)  # Increase font size of title

    plt.xticks(fontsize=tick_fontsize, rotation=45, ha='right')  # Rotate x-ticks to avoid overlap
    plt.yticks(fontsize=tick_fontsize)  # Decrease font size of y-ticks (class labels)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')  # Use bbox_inches to ensure everything fits
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curve(labels, preds, num_classes, output_path=None):
    
    # Assuming preds is a list of numpy.ndarrays
    preds_array = np.array(preds)  # Convert list of numpy.ndarrays to a single numpy.ndarray
    preds_tensor = torch.tensor(preds_array)  # Convert numpy.ndarray to torch.Tensor
    print(f"Predictions tensor shape: {preds_tensor.shape}")
    
    if preds_tensor.dim() != 2:
        raise ValueError("Predictions tensor should be 2D with shape (batch_size, num_classes).")

    # Convert labels to one-hot encoding
    labels_one_hot = np.eye(num_classes)[labels]
    
    # Apply softmax to get class probabilities
    preds_prob = torch.softmax(preds_tensor, dim=1).numpy()
    print(f"Predictions probabilities shape: {preds_prob.shape}")
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels_one_hot.ravel(), preds_prob.ravel())
    auc_roc = roc_auc_score(labels_one_hot, preds_prob, multi_class='ovr')

    # Plot ROC curve
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
    # Convert predictions to numpy array
    preds = np.array(preds)
    labels_one_hot = np.eye(num_classes)[labels]

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_one_hot[:, i], preds[:, i])
        auc_roc = roc_auc_score(labels_one_hot[:, i], preds[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {auc_roc:.2f})')

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
def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def generate_metrics_report(y_true, y_pred, class_columns, output_path):
    metrics_report = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("y_true shape:", y_true.shape)

    if len(y_true.shape) == 1:
        # If it's already a 1D array of class indices, use it directly
        y_true_classes = y_true
    elif len(y_true.shape) == 2:
        # If it's one-hot encoded, get class indices
        y_true_classes = np.argmax(y_true, axis=1)
    else:
        print("Unexpected shape for y_true:", y_true.shape)
        return metrics_report  # or raise an error

    if y_true.shape[0] == 0:
        print("No true labels available for metrics calculation.")
        metrics_report['predictions'] = np.argmax(y_pred, axis=1).tolist()  # Save only predictions
        return metrics_report

    print("Shape of y_true:", y_true.shape)  

    # Ensure y_pred is processed similarly
    y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else (y_pred >= 0.5).astype(int)

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


if __name__ == "__main__":
    val_dir = 'validation'
    batch_size = 32

    val_loader, val_dataset = create_val_dataloader(val_dir, batch_size=batch_size)
    model =MobileNetV3ClassifierWithAttention(num_classes=10)

    # Load the trained model weights
    trained_model_path = 'atten_focal_equal/results_mobv3_atten_focal/model.pth'  # Adjust the path if needed
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    print(f"Loaded trained model from {trained_model_path}")

    # Define the loss function (consistent with training)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # Perform validation
    validate_model(val_loader, model, criterion)

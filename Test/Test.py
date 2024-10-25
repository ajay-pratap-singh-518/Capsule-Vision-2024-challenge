import torch
import pandas as pd
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, recall_score, f1_score, balanced_accuracy_score
import os
import cv2
import json
from glob import glob
import torch.nn.functional as F
class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# Define device 
class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = os.listdir(folder)  # Load all image filenames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder, self.images[idx])
        image = Image.open(img_name).convert('RGB')  # Convert image to RGB if necessary
        if self.transform:
            image = self.transform(image)
        return image, self.images[idx]  # Return the image and its filename

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.size(0) != targets.size(0):
            raise ValueError("Shape mismatch between inputs and targets")# Ensure inputs and targets have the same shape
        p = F.softmax(inputs, dim=1)
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

# Define data loaders for training and validation
class attention(nn.Module):
    def __init__(self, in_channels, attention_size):
        super(attention, self).__init__()
        self.query = nn.Conv2d(in_channels, attention_size, kernel_size=1)
        self.key = nn.Conv2d(in_channels, attention_size, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, c, h, w = x.size()
        query = self.query(x).view(batch_size, -1, h * w)   # Create query, key, and value tensors
        key = self.key(x).view(batch_size, -1, h * w)  
        value = self.value(x).view(batch_size, c, h * w)  
        attention_scores = torch.bmm(query.transpose(1, 2), key)   # Compute attention scores
        attention_scores = attention_scores / (h * w) ** 0.5  
        attention_map = self.softmax(attention_scores)   # Apply softmax to get attention map
        out = torch.bmm(value, attention_map.transpose(1, 2))   # Weighted sum of value tensor
        out = out.view(batch_size, c, h, w)  
        out = out + x # Add residual connection
        return out
class mobilenetv3(nn.Module):
    def __init__(self, num_classes=10, attention_size=64):
        super(mobilenetv3, self).__init__()
        self.mobilenet_v3 = models.mobilenet_v3_large(weights='DEFAULT') # Load the pre-trained MobileNetV3 Large model
        in_features = self.mobilenet_v3.classifier[0].in_features# Get the number of input features for the classifier layer
        self.self_attention = SelfAttention(in_channels=in_features, attention_size=attention_size)# Add self-attention block after certain layers
        self.mobilenet_v3.classifier = nn.Sequential( # Replace the classifier layer to match the number of classes
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.mobilenet_v3.features(x)# Pass through MobileNetV3 backbone
        x = self.self_attention(x) # Apply self-attention block
        x = F.adaptive_avg_pool2d(x, (1, 1))# Global Average Pooling 
        x = torch.flatten(x, 1)
        x = self.mobilenet_v3.classifier(x) # Classification layer
        return x


def generate_gradcam(model, input_tensor, target_class, final_layer_name='mobilenet_v3.features.16.2'):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    final_layer = dict([*model.named_modules()])[final_layer_name]
    forward_hook_handle = final_layer.register_forward_hook(forward_hook)
    backward_hook_handle = final_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()

    one_hot_output = torch.zeros(output.size(), device=output.device)
    one_hot_output[0, target_class] = 1
    output.backward(gradient=one_hot_output)

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * act, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    else:
        cam = np.zeros_like(cam)

    cam = cv2.resize(cam, (224, 224))

    forward_hook_handle.remove()
    backward_hook_handle.remove()

    return cam

def overlay_heatmap_on_image(image_np, heatmap):
    # Ensure heatmap is a single-channel image (if it's multi-channel, reduce it)
    if heatmap.ndim == 3:  # If it's a multi-channel heatmap, convert it to single channel
        heatmap = heatmap.mean(axis=2)

    # Normalize the heatmap to range [0, 1]
    if heatmap.max() != heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalizing
    else:
        heatmap = np.zeros_like(heatmap)  # or set to some other default behavior
 # Normalizing
    # Scale to 0-255 and convert to uint8
    heatmap = np.nan_to_num(heatmap)  # Replace NaNs with zero and infinities with large finite numbers
    heatmap = np.clip(heatmap, 0, 1)  # Ensure values are in [0, 1] range
    heatmap = (heatmap * 255).astype(np.uint8)
    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map
    # Resize heatmap to match the input image size if needed
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    # Combine the heatmap with the original image
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return overlay

# Define data loaders for training and validation
def create_test_dataloader(test_dir, batch_size=32):
    # Define the transform for the test dataset
    test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
    
    # Create the test dataset
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    print(f"Number of images in test dataset: {len(test_dataset)}")
    
    # Create the test dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader, test_dataset
def save_predictions_to_excel(image_paths, y_pred, output_path):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    y_pred_classes = np.argmax(y_pred, axis=1)
    predicted_class_names = [class_columns[i] for i in y_pred_classes]
    df_prob = pd.DataFrame(y_pred, columns=class_columns)
    df_prob.insert(0, 'image_path', image_paths)
    df_class = pd.DataFrame({'image_path': image_paths, 'predicted_class': predicted_class_names})
    df_merged = pd.merge(df_prob, df_class, on='image_path')
    df_merged.to_excel(output_path, index=False)


def calculate_specificity(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity

def generate_metrics_report(y_true, y_pred, class_columns):
    metrics_report = {}
    
    if y_true.shape[0] == 0:
        print("No true labels available for metrics calculation.")
        metrics_report['predictions'] = np.argmax(y_pred, axis=1).tolist()  # Save only predictions
        return metrics_report

    print("Shape of y_true:", y_true.shape)  
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    class_report = classification_report(y_true_classes, y_pred_classes, target_names=class_columns, output_dict=True, zero_division=0)
    
    auc_roc_scores = {}
    for i, class_name in enumerate(class_columns):
        try:
            auc_roc_scores[class_name] = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc_roc_scores[class_name] = 0.0  
    
    mean_auc_roc = np.mean(list(auc_roc_scores.values()))
    auc_roc_scores['mean_auc'] = mean_auc_roc
    
    specificity_scores = {}
    for i, class_name in enumerate(class_columns):
        specificity_scores[class_name] = calculate_specificity(y_true[:, i], (y_pred[:, i] >= 0.5).astype(int))  
    
    mean_specificity = np.mean(list(specificity_scores.values()))
    specificity_scores['mean_specificity'] = mean_specificity
    
    average_precision_scores = {}
    for i, class_name in enumerate(class_columns):
        try:
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            average_precision_scores[class_name] = auc(recall, precision)
        except ValueError:
            average_precision_scores[class_name] = 0.0  
    
    mean_average_precision = np.mean(list(average_precision_scores.values()))
    average_precision_scores['mean_average_precision'] = mean_average_precision
    
    sensitivity_scores = {}
    for i, class_name in enumerate(class_columns):
        try:
            sensitivity_scores[class_name] = recall_score(y_true[:, i], (y_pred[:, i] >= 0.5).astype(int), zero_division=0)
        except ValueError:
            sensitivity_scores[class_name] = 0.0  
    
    mean_sensitivity = np.mean(list(sensitivity_scores.values()))
    sensitivity_scores['mean_sensitivity'] = mean_sensitivity
    
    f1_scores = {}
    for i, class_name in enumerate(class_columns):
        try:
            f1_scores[class_name] = f1_score(y_true[:, i], (y_pred[:, i] >= 0.5).astype(int), zero_division=0)
        except ValueError:
            f1_scores[class_name] = 0.0  
    
    mean_f1_score = np.mean(list(f1_scores.values()))
    f1_scores['mean_f1_score'] = mean_f1_score
    
    balanced_accuracy_scores = balanced_accuracy_score(y_true_classes, y_pred_classes) if y_true.shape[0] > 0 else 0.0

    metrics_report.update(class_report)
    metrics_report['auc_roc_scores'] = auc_roc_scores
    metrics_report['specificity_scores'] = specificity_scores
    metrics_report['average_precision_scores'] = average_precision_scores
    metrics_report['sensitivity_scores'] = sensitivity_scores
    metrics_report['f1_scores'] = f1_scores
    metrics_report['mean_auc'] = mean_auc_roc
    metrics_report['mean_specificity'] = mean_specificity
    metrics_report['mean_average_precision'] = mean_average_precision
    metrics_report['mean_sensitivity'] = mean_sensitivity
    metrics_report['mean_f1_score'] = mean_f1_score
    metrics_report['balanced_accuracy'] = balanced_accuracy_scores
    
    metrics_report_json = json.dumps(metrics_report, indent=4)
    return metrics_report_json


# Define the test directory and output folder
test_dir = 'Testing set/Testing set/Images'
output_folder = 'test_mobv3_atten_focal'
os.makedirs(output_folder, exist_ok=True)

# Initialize the model, criterion, and optimizer
num_classes = 10
model = mobilenetv3(num_classes=10)
for name, layer in model.named_modules():
        print(name)
model.load_state_dict(torch.load('atten/results_mobv3_atten_focal/model.pth'))  # Load the trained model weights
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Create data loader for the test dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to your desired input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])# Convert PIL image to tensor
])

test_folder="test_new/Images"
test_dataset = CustomTestDataset(test_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to evaluate the model on the test dataset
def evaluate_model_on_test(test_dir, gradcam_save_dir, model_path, transform):
    os.makedirs(gradcam_save_dir, exist_ok=True)

    
    test_dataset = CustomTestDataset(test_dir, transform=transform)# Load the test dataset
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    
    model = mobilenetv3(num_classes=len(class_columns))# Load the model
    
    model.load_state_dict(torch.load(model_path), strict=False)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    
    test_preds = []
    test_probs = []
    y_true = []  # List to collect true labels
    filenames = []  # Initialize a list to store filenames
    
    with torch.set_grad_enabled(True):
        for inputs, batch_filenames in test_loader:  # Ensure your dataset returns labels
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            test_preds.extend(predicted.cpu().numpy())
            test_probs.extend(F.softmax(outputs, dim=1).cpu().detach().numpy().tolist())
            #print(test_probs)
            filenames.extend(batch_filenames)  # Collect filenames

            # Generate Grad-CAM images as before...
            for i in range(inputs.size(0)):
                input_tensor = inputs[i:i+1]
                target_class = predicted[i].item()
                
                # Generate Grad-CAM heatmap
                heatmap = generate_gradcam(model, input_tensor, target_class)
                
                # Convert input tensor to image
                image_np = inputs[i].cpu().numpy().transpose(1, 2, 0)
                image_np = (image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                image_np = np.clip(image_np, 0, 1) * 255.0
                image_np = image_np.astype(np.uint8)

                # Overlay the heatmap on the image
                overlay = overlay_heatmap_on_image(image_np, heatmap)
                
                # Prepare the save path
                image_name = batch_filenames[i]  # Use batch_filenames
                gradcam_image_path = os.path.join(gradcam_save_dir, f"{class_columns[target_class]}_{image_name}")
                
                # Save the Grad-CAM image
                cv2.imwrite(gradcam_image_path, overlay)
                print(f"Saved Grad-CAM to {gradcam_image_path}")

    return test_preds, test_probs, y_true, filenames  # Return y_true too


test_folder="Testing set/Testing set/Images"
model_path = "atten/results_mobv3_atten_focal/model.pth"               
gradcam_save_dir = "test_mobv3/gradcam/"
print(type(gradcam_save_dir))  # This should print <class 'str'>

# Evaluate the model
test_preds, test_probs, y_true, image_paths = evaluate_model_on_test('Testing set/Testing set/Images', gradcam_save_dir, model_path, transform)

# Save the predictions and probabilities
save_predictions_to_excel(
    image_paths=image_paths,  # Use image_paths here
    y_pred=test_probs,        # y_pred corresponds to test_probs
    output_path=os.path.join(output_folder, 'test_predictions.xlsx')
)

# Call generate_metrics_report with both y_true and y_pred
generate_metrics_report(
    y_true=np.array(y_true),  # Make sure y_true is in the correct format
    y_pred=test_probs,
    class_columns=class_columns  # Pass class_columns here
)

print("Test evaluation completed and results saved.")

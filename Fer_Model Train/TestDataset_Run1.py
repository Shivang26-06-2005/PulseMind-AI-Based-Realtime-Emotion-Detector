import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
IMG_HEIGHT, IMG_WIDTH = 48, 48
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DATASET_PATH = r'D:\DataSet\Emotion_Images\test'  # your test folder
MODEL_PATH = 'fer_model.pth'  # trained model

# -----------------------------
# Data Transform (same as training)
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# Load Test Dataset
# -----------------------------
test_dataset = datasets.ImageFolder(root=TEST_DATASET_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
num_classes = len(test_dataset.classes)
print(f"Detected classes: {test_dataset.classes}")

# -----------------------------
# Define CNN Model (same as training)
# -----------------------------
class FER_CNN(nn.Module):
    def __init__(self, num_classes):
        super(FER_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# Load Trained Model
# -----------------------------
model = FER_CNN(num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# Evaluate on Test Data
# -----------------------------
criterion = nn.CrossEntropyLoss()
all_labels = []
all_preds = []
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

test_loss /= len(test_dataset)
test_accuracy = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

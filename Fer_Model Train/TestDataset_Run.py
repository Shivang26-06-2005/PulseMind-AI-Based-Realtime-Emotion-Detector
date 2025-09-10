import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ---------------------------
# 1. Device setup
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------------------
# 2. Define model (same as finetuned)
# ---------------------------
class FER_CNN(nn.Module):
    def __init__(self, num_classes=7):
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
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------------
# 3. Load model weights
# ---------------------------
MODEL_PATH = r"fer_model_finetuned_v2.pth"
model = FER_CNN(num_classes=7).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print("Model loaded successfully!")

# ---------------------------
# 4. Define test dataset
# ---------------------------
TEST_DIR = r"D:\DataSet\Emotion_Images\test"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

EMOTIONS = test_dataset.classes  # ['Angry', 'Disgust', ...]

# ---------------------------
# 5. Evaluate & collect predictions
# ---------------------------
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# ---------------------------
# 6. Compute accuracy
# ---------------------------
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

overall_acc = (all_preds == all_labels).sum() / len(all_labels)
print(f"Overall Accuracy: {100 * overall_acc:.2f}%\n")

# Per-class accuracy
for i, emotion in enumerate(EMOTIONS):
    idx = all_labels == i
    if np.sum(idx) > 0:
        acc = (all_preds[idx] == all_labels[idx]).sum() / np.sum(idx)
        print(f"{emotion}: {100*acc:.2f}% ({(all_preds[idx] == all_labels[idx]).sum()}/{np.sum(idx)})")
    else:
        print(f"{emotion}: No samples in test set")

# ---------------------------
# 7. Confusion matrix
# ---------------------------
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# ---------------------------
# 8. F1-score and classification report
# ---------------------------
report = classification_report(all_labels, all_preds, target_names=EMOTIONS)
print("\nClassification Report (includes F1-score):")
print(report)

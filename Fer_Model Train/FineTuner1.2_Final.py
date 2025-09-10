import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

# -----------------------------
# Parameters
# -----------------------------
IMG_HEIGHT, IMG_WIDTH = 48, 48
BATCH_SIZE = 64
EPOCHS = 30
DATASET_PATH = r'D:\DataSet\Emotion_Images\train'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAINED_PATH = 'fer_model_finetuned_v2.pth'
NEW_MODEL_PATH = 'fer_model_finetuned_v3.pth'

# -----------------------------
# Data Transformations
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# Load Dataset
# -----------------------------
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
num_classes = len(full_dataset.classes)
print(f"Detected classes: {full_dataset.classes}")

# Split into train + val
total_size = len(full_dataset)
train_size = int(0.85 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Combine train + val for training
combined_dataset = ConcatDataset([train_dataset, val_dataset])

# -----------------------------
# Weighted Sampler (extra weight for Fear & Angry)
# -----------------------------
combined_indices = combined_dataset.datasets[0].indices if isinstance(combined_dataset.datasets[0], torch.utils.data.Subset) else range(len(combined_dataset))
combined_labels = []
for ds in combined_dataset.datasets:
    if isinstance(ds, torch.utils.data.Subset):
        combined_labels += [full_dataset.imgs[i][1] for i in ds.indices]
    else:
        combined_labels += [label for _, label in ds]

class_counts = [0]*num_classes
for label in combined_labels:
    class_counts[label] += 1

base_weights = [1.0 / c if c>0 else 0.0 for c in class_counts]
extra_weights = [2.0 if i in [0,2] else 1.0 for i in range(num_classes)]  # Fear=0, Angry=2
sample_weights = [base_weights[label]*extra_weights[label] for label in combined_labels]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, sampler=sampler)

# -----------------------------
# Model Definition
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
            nn.Linear(128*6*6, 256),
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

model = FER_CNN(num_classes).to(DEVICE)

# -----------------------------
# Load pretrained v2 model
# -----------------------------
pretrained_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
model_dict = model.state_dict()
pretrained_dict = {k:v for k,v in pretrained_dict.items() if 'features' in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print(f"Loaded {PRETRAINED_PATH} for fine-tuning.")

# Unfreeze last conv block
for name, param in model.features.named_parameters():
    if "2" in name:  # last conv block
        param.requires_grad = True

# -----------------------------
# Focal Loss
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        pt = torch.exp(-logp)
        loss = ((1 - pt) ** self.gamma) * logp
        return loss

criterion = FocalLoss()

# -----------------------------
# Optimizer
# -----------------------------
optimizer = optim.Adam([
    {'params': model.classifier.parameters(), 'lr': 1e-4},
    {'params': model.features[6:].parameters(), 'lr': 1e-5}
], weight_decay=1e-5)

# -----------------------------
# Training Loop (train + val only)
# -----------------------------
best_loss = float('inf')
patience, counter = 7, 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    total, correct = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # Early stopping on loss
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        torch.save(model.state_dict(), NEW_MODEL_PATH)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

print(f"Model saved to {NEW_MODEL_PATH}")

# -----------------------------
# Fine-tune FER CNN (v2) with class weighting
# -----------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

# -----------------------------
# Parameters
# -----------------------------
IMG_HEIGHT, IMG_WIDTH = 48, 48
BATCH_SIZE = 64
EPOCHS = 30
DATASET_PATH = r'D:\DataSet\Emotion_Images\train'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAINED_PATH = 'fer_model_finetuned.pth'
NEW_MODEL_PATH = 'fer_model_finetuned_v2.pth'

# -----------------------------
# Data Transformations
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# Load Dataset
# -----------------------------
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
num_classes = len(full_dataset.classes)
print(f"Detected classes: {full_dataset.classes}")

# Split dataset: 70% train, 15% val, 15% test
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# -----------------------------
# Weighted Sampler for imbalanced classes
# -----------------------------
train_indices = train_dataset.indices
train_labels = [full_dataset.imgs[i][1] for i in train_indices]

class_counts = [0] * num_classes
for label in train_labels:
    class_counts[label] += 1

class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
sample_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
# Load pretrained finetuned model
# -----------------------------
pretrained_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'features' in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print(f"Loaded {PRETRAINED_PATH} for further fine-tuning.")

# Optionally freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False  # unfreeze later if needed

# -----------------------------
# Loss, Optimizer, Scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# -----------------------------
# Early stopping
# -----------------------------
best_val_loss = float('inf')
patience = 7
counter = 0

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), NEW_MODEL_PATH)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# -----------------------------
# Test Accuracy
# -----------------------------
model.load_state_dict(torch.load(NEW_MODEL_PATH))
model.eval()
test_correct, test_total = 0, 0
class_correct = [0]*num_classes
class_total = [0]*num_classes
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        for i in range(len(labels)):
            label = labels[i]
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

print(f"Overall Test Accuracy: {test_correct/test_total:.4f}")
for i, cls in enumerate(full_dataset.classes):
    acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"{cls}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.datasets import DatasetFolder
import librosa
import numpy as np

# ================================
# Step 1: Audio Preprocessing
# ================================
SAMPLE_RATE = 16000
N_MELS = 64
MAX_FRAMES = 200  # fixed length for all spectrograms

def load_audio(path):
    try:
        waveform, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"⚠️ Skipping file {path} due to error: {e}")
        return torch.zeros(1, N_MELS, MAX_FRAMES)

    spec = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_mels=N_MELS)
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

    if spec.size(2) < MAX_FRAMES:
        spec = F.pad(spec, (0, MAX_FRAMES - spec.size(2)))
    else:
        spec = spec[:, :, :MAX_FRAMES]

    return spec

# ================================
# Step 2: Load Datasets and Combine
# ================================
oaf_dataset = DatasetFolder(
    root=r"D:\DataSet\Sound Dataset\TESS Toronto emotional speech set data\OAF",
    loader=load_audio,
    extensions=(".wav",)
)

yaf_dataset = DatasetFolder(
    root=r"D:\DataSet\Sound Dataset\TESS Toronto emotional speech set data\YAF",
    loader=load_audio,
    extensions=(".wav",)
)

# Combine both datasets
full_dataset = ConcatDataset([oaf_dataset, yaf_dataset])
class_labels = oaf_dataset.classes
print("Classes:", class_labels)

# Split dataset into train/val/test
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# ================================
# Step 3: CNN-Only Model (1 layer)
# ================================
class CNN_Model(nn.Module):
    def __init__(self, num_classes=len(class_labels)):
        super(CNN_Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32 * (N_MELS // 2) * (MAX_FRAMES // 2), num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.fc(x)
        return out

# ================================
# Step 4: Training with Validation
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
EPOCHS = 20

print("🚀 Training...")

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    train_acc = correct / len(train_dataset)
    train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_acc = val_correct / len(val_dataset)
    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ================================
# Step 5: Test Accuracy
# ================================
model.eval()
correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        correct += (outputs.argmax(1) == labels).sum().item()
test_acc = correct / len(test_dataset)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ================================
# Step 6: Save Model
# ================================
save_path = "cnn_emotion_model.pt"
torch.save(model.state_dict(), save_path)
print(f"💾 Model saved to {save_path}")

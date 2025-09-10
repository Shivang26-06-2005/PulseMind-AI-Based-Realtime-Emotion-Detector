import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===========================
# CONFIG
# ===========================
TESS_PATH = r"D:\DataSet\Sound Dataset\TESS Toronto emotional speech set data"
SAMPLE_RATE = 16000
N_MFCC = 40
BATCH_SIZE = 32
NUM_EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 1. Dataset preparation (fixed mapping)
# ===========================
emotion_map = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "pleasant_surprised": "surprise"
}

files, labels, speakers = [], [], []

for root, dirs, filenames in os.walk(TESS_PATH):
    for file in filenames:
        if file.lower().endswith('.wav'):
            folder_name = os.path.basename(root).lower()
            for key, emotion in emotion_map.items():
                if folder_name.endswith(key):
                    files.append(os.path.join(root, file))
                    labels.append(emotion)
                    speaker_id = folder_name.split('_')[0]  # e.g., OAF or YAF
                    speakers.append(speaker_id.lower())
                    break

print(f"Found {len(files)} audio files across {len(np.unique(labels))} emotions.")

# ===========================
# 2. Feature extraction (safe + debug)
# ===========================
def extract_features(file_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        # ensure non-empty
        if y is None or len(y) == 0:
            raise ValueError("Empty audio")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    except Exception as e:
        print(f"❌ Could not process {file_path}: {e}")
        return None

# extract (may be a bit slow)
X_temp = [extract_features(f) for f in files]

# report failed files
failed_count = sum(1 for x in X_temp if x is None)
if failed_count:
    print(f"⚠️ {failed_count} files failed MFCC extraction and will be skipped.")

# Filter out None values and keep labels/speakers aligned
valid_idx = [i for i, x in enumerate(X_temp) if x is not None]
X = [X_temp[i] for i in valid_idx]
labels = [labels[i] for i in valid_idx]
speakers = [speakers[i] for i in valid_idx]
files = [files[i] for i in valid_idx]

print(f"✅ Successfully processed {len(X)} / {len(X_temp)} audio files.")

# Pad/truncate to same time length (max frames)
max_len = max(x.shape[1] for x in X)
X_padded = np.array([
    np.pad(x, ((0,0),(0,max_len - x.shape[1])), mode='constant')
    for x in X
], dtype=np.float32)

# ===========================
# 3. Labels / speaker arrays
# ===========================
le = LabelEncoder()
y = le.fit_transform(labels)     # global encoding across all emotions
speakers = np.array(speakers)

print("Label classes:", le.classes_)

# ===========================
# 4. Train = OAF, Test = YAF
# ===========================
train_idx = speakers == "oaf"
test_idx = speakers == "yaf"

if train_idx.sum() == 0 or test_idx.sum() == 0:
    raise RuntimeError("Train or test speaker slice is empty. Check that 'oaf' and 'yaf' exist in speaker folder names.")

X_train, X_test = X_padded[train_idx], X_padded[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train samples (OAF): {len(y_train)}, Test samples (YAF): {len(y_test)}")

# ===========================
# 5. PyTorch Dataset
# ===========================
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, n_mfcc, T)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,n_mfcc,T)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EmotionDataset(X_train, y_train)
test_dataset = EmotionDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===========================
# 6. 2-layer CNN + Attention model
# ===========================
class CNN_Attention(nn.Module):
    def __init__(self, input_height, input_width, num_classes, dropout=0.3):
        super(CNN_Attention, self).__init__()

        # --- CNN Feature extractor ---
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # After two pools → (input_height//4, input_width//4)
        # feature_dim = channels * freq_bins_after_pooling
        self.freq_bins_after = input_height // 4
        self.feature_dim = 64 * self.freq_bins_after

        # --- Attention Layer (on time axis) ---
        self.attention_fc = nn.Linear(self.feature_dim, 128)
        self.attention_score = nn.Linear(128, 1)

        # --- Classification head ---
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B,1,H,W)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # shape: (B,64,H/4,W/4)

        B, C, H, T = x.shape  # H is freq after pooling, T is time after pooling
        # Flatten frequency+channels into feature vector per time-step
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, C, H)
        x = x.view(B, T, -1)                    # (B, T, feature_dim)

        # Attention weights
        attn_hidden = torch.tanh(self.attention_fc(x))   # (B, T, 128)
        attn_scores = self.attention_score(attn_hidden)  # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)     # (B, T, 1)

        # Weighted sum over time -> context vector (B, feature_dim)
        context = torch.sum(attn_weights * x, dim=1)

        # Classifier
        out = self.dropout(F.relu(self.fc1(context)))
        out = self.fc2(out)
        return out

# ===========================
# 7. Instantiate model, loss, optimizer
# ===========================
input_height, input_width = X_train.shape[1], X_train.shape[2]
num_classes = len(np.unique(y))

model = CNN_Attention(input_height, input_width, num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(model)
print("Num parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ===========================
# 8. Training loop
# ===========================
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / max(1, len(train_loader))
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

# ===========================
# 9. Evaluation
# ===========================
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"OAF→YAF Test Accuracy: {acc*100:.2f}%")

# Safer classification report (only include labels present in y_test)
unique_test_labels = np.unique(y_test)
target_names = le.inverse_transform(unique_test_labels)
print(classification_report(y_true, y_pred, labels=unique_test_labels,
                            target_names=target_names, zero_division=0))

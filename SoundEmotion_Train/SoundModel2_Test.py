import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===========================
# 1. Load dataset again
# ===========================
TESS_PATH = r"D:\DataSet\Sound Dataset\TESS Toronto emotional speech set data"

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
        if file.endswith('.wav'):
            folder_name = os.path.basename(root).lower()
            for key, emotion in emotion_map.items():
                if folder_name.endswith(key):
                    files.append(os.path.join(root, file))
                    labels.append(emotion)
                    speaker_id = folder_name.split('_')[0]  # OAF or YAF
                    speakers.append(speaker_id.lower())
                    break

print(f"Found {len(files)} audio files across {len(np.unique(labels))} emotions.")

# ===========================
# 2. Feature extraction
# ===========================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfcc

X = [extract_features(f) for f in files]
max_len = max(x.shape[1] for x in X)
X_padded = np.array([np.pad(x, ((0,0),(0,max_len - x.shape[1])), mode='constant') for x in X])

le = LabelEncoder()
y = le.fit_transform(labels)
speakers = np.array(speakers)

# Test set (YAF speaker)
test_idx = speakers == "yaf"
X_test, y_test = X_padded[test_idx], y[test_idx]

print(f"Test samples (YAF): {len(y_test)}")

# ===========================
# 3. Dataset class
# ===========================
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

test_dataset = EmotionDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ===========================
# 4. CNN model (must match training!)
# ===========================
class TwoLayerCNN(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32*(input_height//4)*(input_width//4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ===========================
# 5. Load trained model
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_height, input_width = X_test.shape[1], X_test.shape[2]
num_classes = len(np.unique(y))

model = TwoLayerCNN(input_height, input_width, num_classes).to(device)
model.load_state_dict(torch.load("sound_emotion_model.pth", map_location=device))
model.eval()

# ===========================
# 6. Evaluation
# ===========================
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy (YAF): {acc*100:.2f}%")

unique_test_labels = np.unique(y_test)
target_names = le.inverse_transform(unique_test_labels)
print(classification_report(y_true, y_pred, labels=unique_test_labels,
                            target_names=target_names, zero_division=0))

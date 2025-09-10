
# ===========================
# 1. Dataset preparation (fixed mapping)
# ===========================
TESS_PATH = r"D:\DataSet\Sound Dataset\TESS Toronto emotional speech set data\TESS Toronto emotional speech set data"

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
    return mfcc  # Keep 2D for CNN

X = [extract_features(f) for f in files]
# Pad/truncate to same length (max frames)
max_len = max(x.shape[1] for x in X)
X_padded = np.array([np.pad(x, ((0,0),(0,max_len - x.shape[1])), mode='constant') for x in X])

le = LabelEncoder()
y = le.fit_transform(labels)
speakers = np.array(speakers)

# ===========================
# 3. Train = OAF, Test = YAF
# ===========================
train_idx = speakers == "oaf"
test_idx = speakers == "yaf"

X_train, X_test = X_padded[train_idx], X_padded[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train samples (OAF): {len(y_train)}, Test samples (YAF): {len(y_test)}")

# ===========================
# 4. PyTorch Dataset
# ===========================
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 40, T)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EmotionDataset(X_train, y_train)
test_dataset = EmotionDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ===========================
# 5. CNN + BiLSTM Model
# ===========================
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_height, input_width, num_classes, hidden_size=128, num_layers=1):
        super(CNN_BiLSTM, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # LSTM input size = number of features after CNN along freq dimension
        cnn_out_height = input_height // 4  # two poolings
        self.lstm_input_size = 32 * cnn_out_height

        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.fc1 = nn.Linear(hidden_size*2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # (B, 16, H/2, W/2)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # (B, 32, H/4, W/4)

        # Prepare for LSTM: (B, 32*H/4, W/4)
        x = x.permute(0, 3, 1, 2)  # (B, W/4, 32, H/4)
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # (B, seq_len, features)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, seq_len, hidden*2)
        lstm_out = lstm_out[:, -1, :]  # take last frame

        # Fully connected
        x = self.dropout(self.relu(self.fc1(lstm_out)))
        x = self.fc2(x)
        return x

# ===========================
# 6. Training
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_height, input_width = X_train.shape[1], X_train.shape[2]
num_classes = len(np.unique(y))

model = CNN_BiLSTM(input_height, input_width, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# ===========================
# 7. Evaluation
# ===========================
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"OAF→YAF Test Accuracy: {acc*100:.2f}%")

unique_test_labels = np.unique(y_test)
target_names = le.inverse_tra
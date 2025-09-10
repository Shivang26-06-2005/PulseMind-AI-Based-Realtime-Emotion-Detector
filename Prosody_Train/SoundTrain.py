import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------
# 1. Load CSV and preprocess
# -----------------------
csv_file = r'D:\DataSet\Sound_2Dataset\features_mp3.csv'
df = pd.read_csv(csv_file)

# Features: drop 'file_name'
X = df.drop(['file_name'], axis=1).values

# Labels: replace with your actual label column if available
y = np.random.randint(0, 2, size=(X.shape[0],))  # binary labels for demo

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for CNN+LSTM: (samples, timesteps, features=1)
X = X.reshape((X.shape[0], X.shape[1], 1))

# --- 80% train / 20% test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# -----------------------
# 2. Create PyTorch Dataset
# -----------------------
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -----------------------
# 3. Define CNN+LSTM Model
# -----------------------
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim):
        super(CNN_LSTM, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)  # Binary classification
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channels=1, length)
        x = torch.relu(self.cnn1(x))
        x = self.pool1(x)
        x = torch.relu(self.cnn2(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1)  # LSTM expects (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out)).squeeze()
        return out

model = CNN_LSTM(input_dim=X_train.shape[1])
print(model)

# -----------------------
# 4. Loss and Optimizer
# -----------------------
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# 5. Training Loop
# -----------------------
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# -----------------------
# 6. Evaluation
# -----------------------
model.eval()
with torch.no_grad():
    y_pred = []
    y_true = []
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        y_pred.extend((outputs >= 0.5).int().numpy())
        y_true.extend(batch_y.numpy())
        
acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")

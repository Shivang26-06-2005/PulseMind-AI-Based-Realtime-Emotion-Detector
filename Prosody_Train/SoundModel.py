import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. Feature Extraction
# =========================
def extract_targets(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    # Pitch (F0)
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_vals = [pitches[mags[:, i].argmax(), i] for i in range(pitches.shape[1])]
    pitch_vals = [p for p in pitch_vals if p > 0]
    pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0

    # Intensity
    rms = librosa.feature.rms(y=y)[0]
    intensity = float(np.mean(rms))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    # Pause ratio
    intervals = librosa.effects.split(y, top_db=30)
    pauses = []
    prev_end = 0
    for start, end in intervals:
        pauses.append((prev_end, start))
        prev_end = end
    pause_ratio = float(sum([e - s for s, e in pauses]) / duration) if duration > 0 else 0.0

    # Stress pattern (approx = pitch variance + intensity variance)
    stress = float(np.std(pitch_vals) + np.std(rms))

    return np.array([pitch, intensity, tempo, pause_ratio, stress], dtype=np.float32)


def extract_spectrogram(audio_path, max_len=128):
    y, sr = librosa.load(audio_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0, 0), (0, max_len - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
    return mel_db.astype(np.float32)


# =========================
# 2. Dataset
# =========================
class ProsodyDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp3")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        spec = extract_spectrogram(fpath)
        target = extract_targets(fpath)
        return torch.tensor(spec).unsqueeze(0), torch.tensor(target)


# =========================
# 3. Model (2-CNN + Bi-LSTM)
# =========================
class ProsodyNet(nn.Module):
    def __init__(self, num_targets=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.lstm = nn.LSTM(input_size=32*16, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*2, num_targets)

    def forward(self, x):
        # x: (B,1,64,128)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # (B,16,32,64)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # (B,32,16,32)
        
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C*H)  # (B,seq,feat)

        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last hidden state
        out = self.fc(out)
        return out


# =========================
# 4. Training
# =========================
audio_dir = r"D:\DataSet\Sound_2Dataset\recordings\recordings"

dataset = ProsodyDataset(audio_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProsodyNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

# Save model
torch.save(model.state_dict(), "prosody_net.pth")
print("✅ Training finished, model saved as prosody_net.pth")

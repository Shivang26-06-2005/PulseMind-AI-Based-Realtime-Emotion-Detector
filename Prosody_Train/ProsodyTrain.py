import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. Helper functions
# =========================
def compute_pause_ratio(y, sr):
    frame_length = 1024
    hop_length = 512
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]

    energy_thresh = 0.3 * np.mean(energy)
    zcr_thresh = 0.3 * np.mean(zcr)

    silence_frames = (energy < energy_thresh) & (zcr < zcr_thresh)
    pause_ratio = np.sum(silence_frames) / len(energy) if len(energy) > 0 else 0.0
    return float(pause_ratio)

def compute_stress_index(pitch_vals, rms):
    if len(pitch_vals) < 2 or len(rms) < 2:
        return 0.0
    dpitch = np.diff(pitch_vals)
    drms = np.diff(rms)
    stress_index = np.mean(np.abs(dpitch)) + np.mean(np.abs(drms))
    return float(stress_index)

def extract_targets(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    # Pitch using YIN
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        pitch_vals = f0[f0 > 0]
        pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
    except:
        pitch = 0.0
        pitch_vals = np.array([0.0])

    # Intensity
    rms = librosa.feature.rms(y=y)[0]
    intensity = float(np.mean(rms)) if len(rms) > 0 else 0.0

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    # Pause ratio
    pause_ratio = compute_pause_ratio(y, sr)

    # Stress index
    stress = compute_stress_index(pitch_vals, rms)

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
        self.files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".wav", ".mp3")):
                    self.files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        spec = extract_spectrogram(fpath)
        target = extract_targets(fpath)
        return torch.tensor(spec, dtype=torch.float32).unsqueeze(0), torch.tensor(target, dtype=torch.float32)

# =========================
# 3. Model
# =========================
class ProsodyNet(nn.Module):
    def __init__(self, num_targets=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.lstm = nn.LSTM(input_size=32*16, hidden_size=128, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*2, num_targets)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C*H)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
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

torch.save(model.state_dict(), "prosody_1.pth")
print("✅ Training finished, model saved as prosody_1.pth")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split

# =========================
# 1. Prosody Feature Extraction
# =========================

def extract_prosody_features(audio_path, sr=16000):
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # -------- Pitch --------
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_vals = [pitches[mags[:, i].argmax(), i] for i in range(pitches.shape[1])]
    pitch_vals = np.array([p for p in pitch_vals if p > 0])
    pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
    
    # -------- Intensity (RMS) --------
    rms = librosa.feature.rms(y=y)[0]
    intensity = float(np.mean(rms))
    
    # -------- Tempo --------
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)
    
    # -------- Pause Ratio --------
    intervals = librosa.effects.split(y, top_db=30)
    pauses = []
    prev_end = 0
    for start, end in intervals:
        pauses.append((prev_end, start))
        prev_end = end
    pause_ratio = float(sum([e - s for s, e in pauses]) / duration) if duration > 0 else 0.0
    
    # -------- Stress Pattern (Improved) --------
    pitch_peaks, _ = find_peaks(pitch_vals, prominence=5)
    rms_peaks, _ = find_peaks(rms, prominence=0.01)
    pitch_stress = np.mean(pitch_vals[pitch_peaks]) if len(pitch_peaks) > 0 else 0.0
    rms_stress = np.mean(rms[rms_peaks]) if len(rms_peaks) > 0 else 0.0
    stress = float(pitch_stress + rms_stress)
    
    return np.array([pitch, intensity, tempo, pause_ratio, stress], dtype=np.float32)

# =========================
# 2. Compute normalization stats
# =========================

def compute_prosody_stats(file_list):
    all_features = []
    for f in file_list:
        feats = extract_prosody_features(f)
        all_features.append(feats)
    all_features = np.stack(all_features)
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0) + 1e-8  # avoid division by zero
    return mean, std

# =========================
# 3. Dataset Class
# =========================

class ProsodyDataset(Dataset):
    def __init__(self, file_list, mean=None, std=None, sr=16000, max_len=128):
        self.files = file_list
        self.mean = mean
        self.std = std
        self.sr = sr
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        
        # Mel-spectrogram
        y, sr = librosa.load(fpath, sr=self.sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] < self.max_len:
            mel_db = np.pad(mel_db, ((0,0),(0,self.max_len - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :self.max_len]
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)

        # Prosody features
        target = extract_prosody_features(fpath)
        if self.mean is not None and self.std is not None:
            target = (target - self.mean) / self.std
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return mel_tensor, target_tensor

# =========================
# 4. CNN+LSTM Model
# =========================

class ProsodyNet(nn.Module):
    def __init__(self, num_targets=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.lstm = nn.LSTM(input_size=32*16, hidden_size=128, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*2, num_targets)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        B,C,H,W = x.size()
        x_seq = x.permute(0,3,1,2).contiguous().view(B,W,C*H)
        out,_ = self.lstm(x_seq)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# =========================
# 5. Training with validation and normalization
# =========================

def train_prosody_model(audio_dir, epochs=20, batch_size=4, lr=1e-3, val_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get all files
    all_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".mp3")]
    
    # Split train/validation
    train_files, val_files = train_test_split(all_files, test_size=val_split, random_state=42)
    
    # Compute normalization from training set only
    mean, std = compute_prosody_stats(train_files)
    print("Prosody mean:", mean)
    print("Prosody std:", std)
    
    # Datasets
    train_dataset = ProsodyDataset(train_files, mean=mean, std=std)
    val_dataset = ProsodyDataset(val_files, mean=mean, std=std)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ProsodyNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                out_val = model(X_val)
                loss_val = criterion(out_val, y_val)
                val_loss_total += loss_val.item()
        val_loss = val_loss_total / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': mean,
                'std': std
            }, "prosody_net_best.pth")
            print("✅ Best model saved")
    
    print("Training complete!")

# =========================
# 6. Run training
# =========================

if __name__ == "__main__":
    audio_dir = r"D:\DataSet\Sound_2Dataset\recordings\recordings"  # <-- Update your folder
    train_prosody_model(audio_dir, epochs=20, batch_size=4, lr=1e-3)

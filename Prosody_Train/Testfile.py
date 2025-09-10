import os
import torch
import torch.nn as nn
import librosa
import numpy as np

# =========================
# 1. CNN-LSTM Model (same architecture)
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
        out = self.fc(out[:, -1, :])
        return out

# =========================
# 2. Mel-spectrogram extraction
# =========================
def extract_spectrogram(audio_path, sr=16000, max_len=128):
    y, sr = librosa.load(audio_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0,0),(0,max_len - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,64,max_len)
    return mel_tensor

# =========================
# 3. Testing function
# =========================
def test_model(audio_folder, model_path):
    # Load pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    mean = checkpoint['mean']
    std = checkpoint['std']

    model = ProsodyNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Iterate over all wav files
    files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
    for f in files:
        path = os.path.join(audio_folder, f)
        mel = extract_spectrogram(path).to(device)

        with torch.no_grad():
            pred = model(mel).cpu().numpy().flatten()
            # Denormalize
            pred_real = pred * std + mean

        print(f"\nPredicted prosody features for '{f}':")
        print(f"Pitch (Hz): {pred_real[0]:.2f}")
        print(f"Intensity (RMS fraction): {pred_real[1]:.2f}")
        print(f"Tempo (BPM): {pred_real[2]:.2f}")
        print(f"Pause Ratio: {pred_real[3]:.2f}")
        print(f"Stress: {pred_real[4]:.2f}")

# =========================
# 4. Run testing
# =========================
if __name__ == "__main__":
    audio_folder = r"D:\DataSet\Sound Dataset\TESS Toronto emotional speech set data\YAF\YAF_sad"  # <-- your folder
    model_path = "prosody_net_best.pth"  # <-- pretrained model
    test_model(audio_folder, model_path)

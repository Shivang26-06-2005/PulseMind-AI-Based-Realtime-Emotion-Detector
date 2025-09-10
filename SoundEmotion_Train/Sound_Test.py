import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import sounddevice as sd
from collections import deque

# ================================
# Model Definition (1-layer CNN)
# ================================
N_MELS = 64
MAX_FRAMES = 200
class_labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']  # replace with your classes

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
# Load Model
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Model().to(device)
model.load_state_dict(torch.load("cnn_emotion_model.pt", map_location=device))
model.eval()

# ================================
# Audio Preprocessing
# ================================
SAMPLE_RATE = 16000

def preprocess_audio(waveform):
    spec = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_mels=N_MELS)
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,frames)
    
    if spec.size(3) < MAX_FRAMES:
        spec = F.pad(spec, (0, MAX_FRAMES - spec.size(3)))
    else:
        spec = spec[:, :, :, :MAX_FRAMES]
    return spec

# ================================
# Real-time Continuous Prediction
# ================================
WINDOW_DURATION = 1.0  # seconds
HOP_DURATION = 0.2     # seconds

buffer = deque(maxlen=int(WINDOW_DURATION * SAMPLE_RATE))

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    buffer.extend(indata[:, 0])

def predict_from_buffer():
    if len(buffer) < int(WINDOW_DURATION * SAMPLE_RATE):
        return  # wait until buffer is filled
    waveform = np.array(buffer)
    input_tensor = preprocess_audio(waveform).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_class = outputs.argmax(1).item()
    print(f"🎯 Predicted Emotion: {class_labels[pred_class]}")

# ================================
# Main Loop
# ================================
if __name__ == "__main__":
    print("🎤 Starting real-time emotion detection...")
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback, blocksize=int(HOP_DURATION*SAMPLE_RATE)):
        while True:
            predict_from_buffer()

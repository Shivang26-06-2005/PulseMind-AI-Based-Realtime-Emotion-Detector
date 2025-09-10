import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd

# ===========================
# 1. Model Definition
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
# 2. Recover Input Shape from Checkpoint
# ===========================
checkpoint = torch.load("sound_emotion_model.pth", map_location="cpu")

fc1_weight_shape = checkpoint["fc1.weight"].shape  # [128, ?]
flattened_size = fc1_weight_shape[1]
print("Flattened size in checkpoint:", flattened_size)

input_height = 40  # MFCCs always 40
# Solve for input_width
input_width = (flattened_size // (32 * (input_height//4))) * 4
print("Recovered input_width:", input_width)

# ===========================
# 3. Load Model
# ===========================
emotion_classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoLayerCNN(input_height, input_width, len(emotion_classes)).to(device)
model.load_state_dict(checkpoint)
model.eval()

# ===========================
# 4. Feature Extraction
# ===========================
def extract_features(file_path, max_len, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

def predict_emotion(file_path):
    features = extract_features(file_path, input_width)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()

    return emotion_classes[predicted_idx]

# ===========================
# 5. Predict Folder
# ===========================
def predict_folder(folder_path, output_csv="predicted_emotions.csv"):
    results = []
    for file in os.listdir(folder_path):
        if file.endswith(".mp3"):
            file_path = os.path.join(folder_path, file)
            emotion = predict_emotion(file_path)
            results.append({"file_name": file, "predicted_emotion": emotion})
            print(f"{file} → {emotion}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to {output_csv}")

# ===========================
# 6. Run
# ===========================
predict_folder(r"D:\DataSet\Sound_2Dataset\recordings\recordings")  # 🔹 change to your folder path

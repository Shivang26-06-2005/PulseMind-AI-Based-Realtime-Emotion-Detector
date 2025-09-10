import os
import numpy as np
import librosa
import pandas as pd
import python_speech_features as psf

# --- CONFIG ---
audio_dir = r'D:\DataSet\Sound_2Dataset\recordings\recordings'  # folder with MP3s
output_csv = r'D:\DataSet\Sound_2Dataset\features_mp3.csv'

# --- FEATURE EXTRACTION FUNCTION ---
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)  # librosa can read MP3
    duration = librosa.get_duration(y=y, sr=sr)

    # Pitch using librosa piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_values.append(pitch)
    pitch_values = np.array(pitch_values)
    mean_pitch = pitch_values.mean().item() if len(pitch_values) > 0 else 0.0
    std_pitch = pitch_values.std().item() if len(pitch_values) > 0 else 0.0

    # Intensity / Energy
    rms = librosa.feature.rms(y=y)[0]
    mean_intensity = rms.mean().item()
    std_intensity = rms.std().item()

    # MFCCs
    mfccs = psf.mfcc(y, samplerate=sr, numcep=13, winlen=0.025, winstep=0.01)
    mfcc_mean = [x.item() for x in mfccs.mean(axis=0)]  # ensure each MFCC is float

    # Tempo / Speech rate
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    # Pauses / Silence
    intervals = librosa.effects.split(y, top_db=30)
    pauses = []
    prev_end = 0
    for start, end in intervals:
        pauses.append((prev_end, start))
        prev_end = end
    pause_ratio = float(sum([e-s for s, e in pauses]) / duration) if duration > 0 else 0.0

    # Modularity and confidence
    modularity = std_pitch + std_intensity
    confidence_score = mean_intensity / (pause_ratio + 0.01)

    # Compile feature vector as plain floats
    features = [mean_pitch, std_pitch, mean_intensity, std_intensity,
                tempo, pause_ratio, confidence_score, modularity]
    features.extend(mfcc_mean)  # already floats

    return features

# --- EXTRACT FEATURES FOR ALL MP3 FILES ---
feature_list = []
file_names = []

for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.lower().endswith('.mp3'):
            file_path = os.path.join(root, file)
            print("Processing:", file_path)
            try:
                feats = extract_features(file_path)
                feature_list.append(feats)
                file_names.append(file)
            except Exception as e:
                print(f"Failed to process {file}: {e}")

# --- CREATE DATAFRAME ---
columns = ['mean_pitch','std_pitch','mean_intensity','std_intensity',
           'tempo','pause_ratio','confidence_score','modularity']
columns += [f'mfcc{i}' for i in range(1,14)]

df = pd.DataFrame(feature_list, columns=columns)
df['file_name'] = file_names

# Save CSV with numeric values only
df.to_csv(output_csv, index=False, float_format='%.6f')
print(f"Features saved successfully to {output_csv}")
    
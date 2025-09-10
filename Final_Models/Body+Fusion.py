import cv2
import mediapipe as mp
import math
from collections import deque

import torch
import numpy as np
import librosa
import pyaudio
import threading
import time

from PIL import Image
import torchvision.transforms as transforms
from torch import nn
from scipy.signal import find_peaks
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# 1. MediaPipe Gesture Logic
# =========================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

GESTURE_COUNTERS = {
    "Leaning Forward": 0,
    "Leaning Back": 0,
    "Leaning Left": 0,
    "Leaning Right": 0,
    "Upright Posture": 0,
    "Hand Raised": 0,
    "Hand Lowered": 0,
    "Pointing": 0,
    "Waving": 0,
}

baseline_z = None
wrist_history = {"left": deque(maxlen=5), "right": deque(maxlen=5)}

def get_precise_direction(dx, dy, dz, thresh=0.01):
    directions = []
    if abs(dx) > thresh:
        directions.append("Right" if dx > 0 else "Left")
    if abs(dy) > thresh:
        directions.append("Down" if dy > 0 else "Up")
    if abs(dz) > thresh:
        directions.append("Back" if dz > 0 else "Forward")
    if not directions:
        directions.append("No Movement")
    return "-".join(directions)

# =========================
# 2. Emotion Model (CNN)
# =========================
class FER_CNN(nn.Module):
    def __init__(self, num_classes):
        super(FER_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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

class ProsodyNet(nn.Module):
    def __init__(self, num_targets=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
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
# 3. Feature Extraction Utils
# =========================
def extract_prosody_features(audio_data, sr=16000):
    y = audio_data
    duration = len(y) / sr
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_vals = [pitches[mags[:, i].argmax(), i] for i in range(pitches.shape[1])]
    pitch_vals = np.array([p for p in pitch_vals if p > 0])
    pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
    rms = librosa.feature.rms(y=y)[0]
    intensity = float(np.mean(rms))
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except:
        tempo = 0.0
    intervals = librosa.effects.split(y, top_db=30)
    pauses = []
    prev_end = 0
    for start, end in intervals:
        pauses.append((prev_end, start))
        prev_end = end
    pause_ratio = float(sum([e - s for s, e in pauses]) / duration) if duration > 0 else 0.0
    if len(pitch_vals) > 1:
        pitch_peaks, _ = find_peaks(pitch_vals, prominence=5)
        pitch_stress = np.mean(pitch_vals[pitch_peaks]) if len(pitch_peaks) > 0 else 0.0
    else:
        pitch_stress = 0.0
    if len(rms) > 1:
        rms_peaks, _ = find_peaks(rms, prominence=0.01)
        rms_stress = np.mean(rms[rms_peaks]) if len(rms_peaks) > 0 else 0.0
    else:
        rms_stress = 0.0
    stress = float(pitch_stress + rms_stress)
    return np.array([pitch, intensity, tempo, pause_ratio, stress], dtype=np.float32)

def extract_mfcc_features(audio_data, sr=16000, max_len=92):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# =========================
# 4. Unified Detector Class
# =========================
class UnifiedBehaviorEmotionDetector:
    def __init__(self):
        self.video_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.sound_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.gesture_counters = dict(GESTURE_COUNTERS)
        # Video/pose/audio setup
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.audio = pyaudio.PyAudio()
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.audio_buffer = deque(maxlen=int(16000 * 2))
        try:
            self.stream = self.audio.open(
                format=self.audio_format, channels=self.channels,
                rate=self.rate, input=True, frames_per_buffer=self.chunk,
                stream_callback=self.audio_callback)
            self.stream.start_stream()
        except Exception as e:
            print("⚠ Audio setup failed:", e)
            self.stream = None
        self.video_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # Load models (add exception handling as in your scripts)
        self.video_model = FER_CNN(7).to(DEVICE)
        self.video_model.eval()
        self.sound_model = TwoLayerCNN(40, 92, 7).to(DEVICE)
        self.sound_model.eval()
        self.prosody_model = ProsodyNet().to(DEVICE)
        self.prosody_model.eval()
        try:
            self.bart_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
            self.bart_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
        except:
            self.bart_tokenizer = None
            self.bart_model = None
        # Feedback/state
        self.current_results = {
            'video_emotion': 'Neutral',
            'sound_emotion': 'neutral',
            'prosody_features': [0.0]*5,
            'feedback_scores': {
                'engagement': 0.5,
                'confidence': 0.5,
                'nervousness': 0.5,
                'positivity': 0.5
            },
            'gesture_summary': '',
            'posture': 'Upright'
        }

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)
        return (None, pyaudio.paContinue)

    def detect_pose_and_gesture(self, image, results):
        gesture_msg = []
        # Extract pose landmarks and counters as in BodyGesture2.py
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            l_sh, r_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_hip, r_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            mid_sh_x = (l_sh.x + r_sh.x)/2
            mid_sh_y = (l_sh.y + r_sh.y)/2
            mid_sh_z = (l_sh.z + r_sh.z)/2
            mid_hip_x = (l_hip.x + r_hip.x)/2
            mid_hip_y = (l_hip.y + r_hip.y)/2
            mid_hip_z = (l_hip.z + r_hip.z)/2
            global baseline_z
            if baseline_z is None:
                baseline_z = (mid_sh_z + mid_hip_z)/2
            dx = mid_hip_x - mid_sh_x
            dz = ((mid_hip_z + mid_sh_z)/2) - baseline_z
            lean_thresh_z = 0.03
            lean_thresh_x = 0.03
            posture = "Upright"
            if dz < -lean_thresh_z:
                posture = "Leaning Forward"
            elif dz > lean_thresh_z:
                posture = "Leaning Back"
            elif dx > lean_thresh_x:
                posture = "Leaning Right"
            elif dx < -lean_thresh_x:
                posture = "Leaning Left"
            self.current_results['posture'] = posture
            # Gestures, hand raised/lowered, pointing, waving (add counter logic)
            l_wrist, r_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            if l_wrist.y < l_sh.y or r_wrist.y < r_sh.y:
                gesture_msg.append("Hand Raised")
            if l_wrist.y >= l_sh.y and r_wrist.y >= r_sh.y:
                gesture_msg.append("Hand Lowered")
            l_elbow, r_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            l_angle = math.degrees(math.atan2(l_wrist.y - l_elbow.y, l_wrist.x - l_elbow.x))
            r_angle = math.degrees(math.atan2(r_wrist.y - r_elbow.y, r_wrist.x - r_elbow.x))
            if abs(l_angle) < 30 or abs(r_angle) < 30:
                gesture_msg.append("Pointing")
            wrist_history["left"].append(l_wrist.x)
            wrist_history["right"].append(r_wrist.x)
            if len(wrist_history["left"])==5 and max(wrist_history["left"]) - min(wrist_history["left"])>0.05:
                gesture_msg.append("Waving (Left)")
            if len(wrist_history["right"])==5 and max(wrist_history["right"]) - min(wrist_history["right"])>0.05:
                gesture_msg.append("Waving (Right)")
            self.current_results['gesture_summary'] = "; ".join(gesture_msg) if gesture_msg else "No distinct gesture"

    def detect_video_emotion(self, frame):
        # Detect face, predict emotion
        small_frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240
            x, y, w, h = [int(val * scale_x if i%2==0 else val * scale_y) for i, val in enumerate(faces[0])]
            face_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_pil = Image.fromarray(face_roi)
            face_tensor = self.video_transform(face_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = self.video_model(face_tensor)
                pred = torch.argmax(output, 1).item()
                emotion = self.video_emotions[pred]
                self.current_results['video_emotion'] = emotion
        else:
            self.current_results['video_emotion'] = "Neutral"

    def detect_sound_emotion(self):
        if len(self.audio_buffer) < 8000: return
        audio_data = np.array(list(self.audio_buffer))
        mfcc = extract_mfcc_features(audio_data)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.sound_model(mfcc_tensor)
            pred = torch.argmax(output, 1).item()
            emotion = self.sound_emotions[pred]
            self.current_results['sound_emotion'] = emotion

    def detect_prosody_features(self):
        if len(self.audio_buffer) < 8000: return
        audio_data = np.array(list(self.audio_buffer))
        prosody_feats = extract_prosody_features(audio_data)
        self.current_results['prosody_features'] = prosody_feats.tolist()

    def analyze_with_bart(self, emotion_text):
        if self.bart_tokenizer is None or self.bart_model is None:
            return {
                'engagement': 0.5,
                'confidence': 0.5,
                'nervousness': 0.5,
                'positivity': 0.5
            }
        feedback_aspects = {
            'engagement': f"The person shows {emotion_text} emotion, indicating engagement level",
            'confidence': f"The person shows {emotion_text} emotion, indicating confidence level",
            'nervousness': f"The person shows {emotion_text} emotion, indicating nervousness level",
            'positivity': f"The person shows {emotion_text} emotion, indicating positivity level"
        }
        scores = {}
        for aspect, text in feedback_aspects.items():
            try:
                hypothesis = f"This shows high {aspect}"
                inputs = self.bart_tokenizer(text, hypothesis, return_tensors='pt', truncation=True)
                with torch.no_grad():
                    outputs = self.bart_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                score = probs[0][2].item()
                scores[aspect] = score
            except:
                scores[aspect] = 0.5
        return scores

    def synthesize_text_feedback(self):
        posture = self.current_results['posture']
        gesture = self.current_results['gesture_summary']
        video_emotion = self.current_results['video_emotion']
        sound_emotion = self.current_results['sound_emotion']
        prosody = self.current_results['prosody_features']
        feedback_scores = self.current_results['feedback_scores']
        feedback_str = (f"Posture: {posture}. Gesture: {gesture}. "
                        f"Facial emotion detected: {video_emotion}. "
                        f"Sound emotion: {sound_emotion}."
                        f"Prosody features: Pitch {prosody[0]:.2f}, Intensity {prosody[1]:.2f}, Tempo {prosody[2]:.2f}, PauseRatio {prosody[3]:.2f}, Stress {prosody[4]:.2f}. "
                        f"Estimated feedback: Engagement {feedback_scores['engagement']:.2f}, Confidence {feedback_scores['confidence']:.2f}, Nervousness {feedback_scores['nervousness']:.2f}, Positivity {feedback_scores['positivity']:.2f}.")
        return feedback_str

    def run(self):
        print("Unified Behavior & Emotion Detection. Press 'q' to quit.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                # MediaPipe pose detection
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                self.detect_pose_and_gesture(frame, results)
                # Deep emotion detection (face, audio, prosody)
                self.detect_video_emotion(frame)
                self.detect_sound_emotion()
                self.detect_prosody_features()
                # BART feedback
                combined_text = f"{self.current_results['video_emotion']} and {self.current_results['sound_emotion']}"
                self.current_results['feedback_scores'] = self.analyze_with_bart(combined_text)
                # Display fused feedback
                fused_text = self.synthesize_text_feedback()
                cv2.putText(frame, fused_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                cv2.imshow('Unified Behavioral Feedback', frame)
                print(fused_text)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.cap.release()
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.audio.terminate()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = UnifiedBehaviorEmotionDetector()
        detector.run()
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

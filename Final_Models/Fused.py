"""
fusion_realtime_analysis_with_opencv_optimized.py  
Realtime fusion script with OpenCV video emotion + Mediapipe pose, audio emotion, prosody, and feedback.  
Optimized for FPS and full feature display with face bounding boxes.
"""

from __future__ import annotations
import time
import threading
import queue
import math
import json
import os
import sys
from typing import Optional, Tuple
from collections import deque

import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as T
import sounddevice as sd
import librosa
from transformers import pipeline

# ---------------------------
# Config / file paths
# ---------------------------
FER_MODEL_PATH = 'fer_model_finetuned_v2.pth'
SOUND_MODEL_PATH = 'sound_emotion_model.pth'
PROSODY_MODEL_PATH = 'prosody_net_best.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VIDEO_SOURCE = 0
SHOW_WINDOW = True

AUDIO_SR = 16000
AUDIO_CHUNK_SECONDS = 2.0
AUDIO_CHANNELS = 1

LOG_PATH = 'fusion_output.log'
FPS_LOG_INTERVAL = 1.0

FER_SKIP_FRAMES = 2  # skip frames for FER to optimize FPS

with open(LOG_PATH, 'w') as f:
    f.write('')

log_lock = threading.Lock()
def log(obj):
    s = json.dumps(obj, default=str, ensure_ascii=False)
    with log_lock:
        print(s)
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(s + '\n')

# ---------------------------
# FER model
# ---------------------------
class FER_CNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super(FER_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

fer_classes = ['angry','disgust','fear','happy','neutral','sad','surprise']
fer_model: Optional[nn.Module] = None
if os.path.exists(FER_MODEL_PATH):
    try:
        tmp = FER_CNN(len(fer_classes))
        state = torch.load(FER_MODEL_PATH, map_location=DEVICE)
        if isinstance(state, dict) and 'model_state_dict' in state:
            tmp.load_state_dict(state['model_state_dict'])
        elif isinstance(state, dict):
            tmp.load_state_dict(state)
        else:
            tmp = state
        tmp.to(DEVICE).eval()
        fer_model = tmp
        log({'info': 'FER model loaded', 'path': FER_MODEL_PATH})
    except Exception as e:
        log({'warning': 'Failed loading FER model', 'error': str(e)})
else:
    log({'warning': 'FER model file not found', 'expected_path': FER_MODEL_PATH})

fer_transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=1),
    T.Resize((48,48)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# ---------------------------
# Sound model
# ---------------------------
class TwoLayerCNN(nn.Module):
    def __init__(self, input_height: int, input_width: int, num_classes: int):
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

sound_model: Optional[nn.Module] = None
if os.path.exists(SOUND_MODEL_PATH):
    try:
        input_h, input_w = 64, 128
        tmp = TwoLayerCNN(input_h, input_w, len(fer_classes))
        state = torch.load(SOUND_MODEL_PATH, map_location=DEVICE)
        if isinstance(state, dict) and 'model_state_dict' in state:
            tmp.load_state_dict(state['model_state_dict'])
        elif isinstance(state, dict):
            tmp.load_state_dict(state)
        else:
            tmp = state
        tmp.to(DEVICE).eval()
        sound_model = tmp
        log({'info':'Sound model loaded','path': SOUND_MODEL_PATH})
    except Exception as e:
        log({'warning':'Failed loading sound model','error':str(e)})
else:
    log({'warning':'Sound model file not found','expected_path':SOUND_MODEL_PATH})

# ---------------------------
# Prosody model
# ---------------------------
class ProsodyNet(nn.Module):
    def __init__(self, num_targets: int = 5):
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
        out = out.mean(dim=1)
        out = self.fc(out)
        return out

prosody_model: Optional[nn.Module] = None
if os.path.exists(PROSODY_MODEL_PATH):
    try:
        pm = ProsodyNet()
        st = torch.load(PROSODY_MODEL_PATH, map_location=DEVICE)
        if isinstance(st, dict) and 'model_state_dict' in st:
            pm.load_state_dict(st['model_state_dict'])
        elif isinstance(st, dict):
            pm.load_state_dict(st)
        else:
            pm = st
        pm.to(DEVICE).eval()
        prosody_model = pm
        log({'info':'Prosody model loaded','path': PROSODY_MODEL_PATH})
    except Exception as e:
        log({'warning':'Failed loading prosody model','error':str(e)})
else:
    log({'warning':'Prosody model not found','expected_path':PROSODY_MODEL_PATH})

# ---------------------------
# Feedback
# ---------------------------
FEEDBACK_ENCODER = "facebook/bart-large-mnli"
try:
    feedback_classifier = pipeline("zero-shot-classification", model=FEEDBACK_ENCODER, device=0 if torch.cuda.is_available() else -1)
    log({'info': 'Feedback encoder loaded', 'model': FEEDBACK_ENCODER})
except Exception as e:
    feedback_classifier = None
    log({'warning':'Failed loading feedback encoder', 'error': str(e)})

FEEDBACK_MAP = {
    'angry': 'Take slow deep breaths and lower your tone. Consider reducing speed and softening facial tension.',
    'disgust': 'Check for discomfort — adjust posture or tone. Rephrase any harsh language.',
    'fear': 'You may appear uncertain. Ground your voice and maintain steady eye contact.',
    'happy': 'Great — positive expressions encourage engagement. Keep it natural.',
    'neutral': 'Neutral stance. Add variation in prosody and gestures to increase engagement.',
    'sad': 'You might appear low energy. Increase vocal energy and openness in posture.',
    'surprise': 'Surprise can be engaging; ensure it matches context and isn\'t confusing.'
}

# ---------------------------
# Mediapipe
# ---------------------------
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------------
# Audio capture
# ---------------------------
audio_q = queue.Queue()
recent_audio_buffer = deque(maxlen=int(AUDIO_SR*10))

def audio_callback(indata, frames, time_info, status):
    if status:
        log({'audio_status': str(status)})
    audio_q.put(indata.copy())

def start_audio_stream():
    try:
        stream = sd.InputStream(channels=AUDIO_CHANNELS, samplerate=AUDIO_SR, callback=audio_callback)
        stream.start()
        log({'info':'Audio stream started','samplerate':AUDIO_SR})
        return stream
    except Exception as e:
        log({'warning':'Failed to start audio stream','error':str(e)})
        return None

def extract_mel_spectrogram(y: np.ndarray, sr: int=AUDIO_SR, n_mels: int=64, n_fft: int=512, hop_length: int=160) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

# ---------------------------
# Inference helpers
# ---------------------------
def infer_fer_from_frame(frame_bgr: np.ndarray) -> Tuple[str,float,Optional[Tuple[int,int,int,int]]]:
    """
    Returns: emotion label, confidence score, bounding box (x1,y1,x2,y2)
    """
    if fer_model is None:
        return ('unknown', 0.0, None)
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    if not results.detections:
        return ('no_face', 0.0, None)
    h,w = frame_bgr.shape[:2]
    best = None
    best_area = 0
    for det in results.detections:
        bb = det.location_data.relative_bounding_box
        x = int(bb.xmin * w)
        y = int(bb.ymin * h)
        bw = int(bb.width * w)
        bh = int(bb.height * h)
        area = bw*bh
        if area > best_area:
            best_area = area
            best = (x,y,bw,bh)
    if best is None:
        return ('no_face',0.0,None)
    x,y,bw,bh = best
    pad = int(0.2*max(bw,bh))
    x1 = max(0, x-pad)
    y1 = max(0, y-pad)
    x2 = min(w, x+bw+pad)
    y2 = min(h, y+bh+pad)
    face = frame_bgr[y1:y2, x1:x2]
    try:
        tensor = fer_transform(face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = fer_model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            return (fer_classes[idx], float(probs[idx]), (x1,y1,x2,y2))
    except Exception as e:
        log({'warning':'FER inference failed','error':str(e)})
        return ('error',0.0,(x1,y1,x2,y2))

# ---------------------------
# Audio worker
# ---------------------------
def audio_worker(stop_event: threading.Event):
    cur_segment = []
    while not stop_event.is_set():
        try:
            block = audio_q.get(timeout=0.5)
            block = block.flatten()
            recent_audio_buffer.extend(block.tolist())
            cur_segment.extend(block.tolist())
            if len(cur_segment) >= int(AUDIO_SR*AUDIO_CHUNK_SECONDS):
                seg = np.array(cur_segment[:int(AUDIO_SR*AUDIO_CHUNK_SECONDS)])
                cur_segment = cur_segment[int(AUDIO_SR*AUDIO_CHUNK_SECONDS):]
                mel = extract_mel_spectrogram(seg, sr=AUDIO_SR)
                mel_norm = (mel - mel.mean()) / (1e-6 + mel.std())
                mel_tensor = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
                if sound_model is not None:
                    try:
                        with torch.no_grad():
                            out = sound_model(mel_tensor)
                            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
                            idx = int(np.argmax(probs))
                            emotion = fer_classes[idx]
                            score = float(probs[idx])
                            log({'audio_emotion':emotion,'score':score})
                    except Exception as e:
                        log({'warning':'Sound model inference failed','error':str(e)})
                if prosody_model is not None:
                    try:
                        inp = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
                        with torch.no_grad():
                            p_out = prosody_model(inp).cpu().numpy()[0].tolist()
                        log({'prosody':p_out})
                    except Exception as e:
                        log({'warning':'Prosody inference failed','error':str(e)})
        except queue.Empty:
            continue

# ---------------------------
# Pose summarizer
# ---------------------------
def summarize_pose(landmarks) -> dict:
    if landmarks is None:
        return {}
    res = {}
    try:
        nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        hip_y = (left_hip.y + right_hip.y)/2.0
        res['leaning_forward'] = nose.y < hip_y - 0.05
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        dx = abs(left_wrist.x - right_wrist.x)
        dy = abs(left_wrist.y - right_wrist.y)
        res['hands_folded'] = (dx < 0.08 and dy < 0.08)
    except Exception:
        pass
    return res

# ---------------------------
# Feedback generator
# ---------------------------
def generate_feedback(detected_emotions: dict, pose_summary: dict) -> str:
    primary = None
    best_score = 0.0
    for k,v in detected_emotions.items():
        if v > best_score:
            best_score = v
            primary = k
    if primary is None or primary in ('no_face','unknown','error'):
        return 'No reliable face/emotion detected. Check camera placement.'
    if feedback_classifier is not None:
        candidate_feedbacks = list(FEEDBACK_MAP.values())
        hypothesis = f"The best feedback for someone feeling {primary} is:"
        try:
            result = feedback_classifier(hypothesis, candidate_feedbacks, multi_label=False)
            fb = result["labels"][0]
        except Exception as e:
            log({'warning': 'Feedback classifier failed', 'error': str(e)})
            fb = FEEDBACK_MAP.get(primary, "No specific feedback available.")
    else:
        fb = FEEDBACK_MAP.get(primary, "No specific feedback available.")
    if pose_summary.get('leaning_forward', False):
        fb += " You appear to lean forward — maintain a relaxed upright posture."
    if pose_summary.get('hands_folded', False):
        fb += " Hands folded can seem closed; open gestures are more engaging."
    return fb

# ---------------------------
# Main loop
# ---------------------------
def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        log({'error':'Failed to open video source', 'source': VIDEO_SOURCE})
        return

    stop_event = threading.Event()
    audio_stream = start_audio_stream()
    audio_thread = threading.Thread(target=audio_worker, args=(stop_event,), daemon=True)
    audio_thread.start()

    frame_count = 0
    last_fps_time = time.time()
    displayed_fps = 0
    recent_audio_emotions = deque(maxlen=5)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log({'warning':'Failed to read frame'})
                break
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= FPS_LOG_INTERVAL:
                displayed_fps = frame_count / (now - last_fps_time)
                last_fps_time = now
                frame_count = 0

            # Video FER
            if frame_count % FER_SKIP_FRAMES == 0:
                video_emotion, video_score, face_bbox = infer_fer_from_frame(frame)
            else:
                video_emotion, video_score, face_bbox = 'skip', 0.0, None

            # Pose
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = pose_detector.process(frame_rgb)
            pose_summary = summarize_pose(pose_res.pose_landmarks) if pose_res.pose_landmarks else {}

            # Audio
            audio_emotion, audio_score = 'none', 0.0
            if recent_audio_buffer:
                try:
                    recent_audio = np.array(list(recent_audio_buffer)[-int(AUDIO_SR*AUDIO_CHUNK_SECONDS):])
                    mel = extract_mel_spectrogram(recent_audio, sr=AUDIO_SR)
                    mel_norm = (mel - mel.mean()) / (1e-6 + mel.std())
                    mel_tensor = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
                    if sound_model is not None:
                        with torch.no_grad():
                            out = sound_model(mel_tensor)
                            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
                            idx = int(np.argmax(probs))
                            audio_emotion = fer_classes[idx]
                            audio_score = float(probs[idx])
                            recent_audio_emotions.append((audio_emotion,audio_score))
                except Exception:
                    pass

            # Prosody
            prosody_values = []
            if prosody_model is not None and recent_audio_buffer:
                try:
                    inp = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
                    with torch.no_grad():
                        prosody_values = prosody_model(inp).cpu().numpy()[0].tolist()
                except Exception:
                    pass

            # Feedback
            detected = {video_emotion: video_score, audio_emotion: audio_score}
            feedback_text = generate_feedback(detected, pose_summary)

            # Display
            if SHOW_WINDOW:
                overlay = frame.copy()
                h,w = frame.shape[:2]
                y = 30
                cv2.putText(overlay, f'Video Emotion: {video_emotion} ({video_score:.2f})', (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                y += 25
                cv2.putText(overlay, f'Audio Emotion: {audio_emotion} ({audio_score:.2f})', (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                y += 25
                cv2.putText(overlay, f'Prosody: {prosody_values}', (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,200), 1)
                y += 25
                for k,v in pose_summary.items():
                    cv2.putText(overlay, f'{k}: {v}', (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
                    y += 25
                # Draw face bounding box
                if face_bbox is not None:
                    x1,y1,x2,y2 = face_bbox
                    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,255), 2)
                    # Put emotion label above box
                    cv2.putText(overlay, f'{video_emotion}', (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                # Feedback text
                for i, line in enumerate([feedback_text[i:i+60] for i in range(0, len(feedback_text), 60)]):
                    cv2.putText(overlay, line, (10, h-60 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(overlay, f'FPS: {displayed_fps:.1f}', (w-120,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0),2)
                cv2.imshow('Fusion Realtime Analysis', overlay)
                key = cv2.waitKey(1)
                if key == 27:
                    break

            log({'timestamp': time.time(), 'video_emotion': video_emotion, 'audio_emotion': audio_emotion,
                 'prosody': prosody_values, 'pose_summary': pose_summary, 'feedback': feedback_text, 'fps': displayed_fps})

    finally:
        stop_event.set()
        try:
            if audio_stream is not None:
                audio_stream.stop()
                audio_stream.close()
        except Exception:
            pass
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        log({'info':'Exited main loop'})

if __name__ == '__main__':
    main()

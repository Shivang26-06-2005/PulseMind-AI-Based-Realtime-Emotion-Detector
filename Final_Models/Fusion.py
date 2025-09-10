import cv2
import torch
import torch.nn as nn
import numpy as np
import librosa
import pyaudio
import threading
import time
from collections import deque
from transformers import BartTokenizer, BartForSequenceClassification, pipeline
import torch.nn.functional as F
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings('ignore')

# ========================================
# Model Definitions (from your training code)
# ========================================

class FER_CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FER_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
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

class TwoLayerCNN(nn.Module):
    def __init__(self, input_height=40, input_width=216, num_classes=7):
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

# ========================================
# Audio Processing Functions
# ========================================

def extract_mfcc_features(audio_data, sr=16000):
    if len(audio_data) == 0:
        return np.zeros((40, 216))
    
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    max_len = 216
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def extract_prosody_features(audio_data, sr=16000):
    if len(audio_data) == 0:
        return np.zeros(5, dtype=np.float32)
    
    duration = len(audio_data) / sr
    
    # Pitch
    pitches, mags = librosa.piptrack(y=audio_data, sr=sr)
    pitch_vals = [pitches[mags[:, i].argmax(), i] for i in range(pitches.shape[1])]
    pitch_vals = np.array([p for p in pitch_vals if p > 0])
    pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
    
    # Intensity (RMS)
    rms = librosa.feature.rms(y=audio_data)[0]
    intensity = float(np.mean(rms))
    
    # Tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        tempo = float(tempo)
    except:
        tempo = 0.0
    
    # Pause Ratio
    try:
        intervals = librosa.effects.split(audio_data, top_db=30)
        pauses = []
        prev_end = 0
        for start, end in intervals:
            pauses.append((prev_end, start))
            prev_end = end
        pause_ratio = float(sum([e - s for s, e in pauses]) / duration) if duration > 0 else 0.0
    except:
        pause_ratio = 0.0
    
    # Stress Pattern
    try:
        if len(pitch_vals) > 0:
            pitch_peaks, _ = find_peaks(pitch_vals, prominence=5)
            pitch_stress = np.mean(pitch_vals[pitch_peaks]) if len(pitch_peaks) > 0 else 0.0
        else:
            pitch_stress = 0.0
        
        rms_peaks, _ = find_peaks(rms, prominence=0.01)
        rms_stress = np.mean(rms[rms_peaks]) if len(rms_peaks) > 0 else 0.0
        stress = float(pitch_stress + rms_stress)
    except:
        stress = 0.0
    
    return np.array([pitch, intensity, tempo, pause_ratio, stress], dtype=np.float32)

def extract_mel_spectrogram(audio_data, sr=16000, max_len=128):
    if len(audio_data) == 0:
        return np.zeros((64, max_len))
    
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0,0),(0,max_len - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
    
    return mel_db

# ========================================
# BART Feedback Generator
# ========================================

class BARTFeedbackGenerator:
    def __init__(self):
        print("Loading BART model for feedback generation...")
        self.classifier = pipeline("zero-shot-classification", 
                                 model="facebook/bart-large-mnli",
                                 device=0 if torch.cuda.is_available() else -1)
        
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.assessment_labels = ['positive', 'negative', 'engaged', 'disengaged', 'confident', 'anxious']
        
        print("BART model loaded successfully!")

    def generate_feedback(self, facial_emotion, voice_emotion, prosody_scores):
        # Create emotion description
        emotion_text = f"Person shows {facial_emotion} facial expression and {voice_emotion} vocal emotion"
        
        # Analyze overall sentiment
        sentiment_result = self.classifier(emotion_text, self.assessment_labels)
        
        # Calculate composite scores
        positivity = self.calculate_positivity(facial_emotion, voice_emotion, prosody_scores)
        engagement = self.calculate_engagement(prosody_scores, facial_emotion)
        confidence = self.calculate_confidence(prosody_scores, voice_emotion)
        neutrality = self.calculate_neutrality(facial_emotion, voice_emotion)
        
        # Generate personalized feedback
        feedback = self.generate_personalized_feedback(
            facial_emotion, voice_emotion, prosody_scores, 
            positivity, engagement, confidence, neutrality
        )
        
        return {
            'feedback': feedback,
            'positivity': positivity,
            'engagement': engagement,
            'confidence': confidence,
            'neutrality': neutrality,
            'dominant_sentiment': sentiment_result['labels'][0],
            'sentiment_scores': dict(zip(sentiment_result['labels'], sentiment_result['scores']))
        }
    
    def calculate_positivity(self, facial_emotion, voice_emotion, prosody_scores):
        positive_emotions = {'happy': 0.9, 'surprise': 0.6, 'neutral': 0.5}
        negative_emotions = {'angry': 0.1, 'sad': 0.2, 'fear': 0.2, 'disgust': 0.1}
        
        face_score = positive_emotions.get(facial_emotion, negative_emotions.get(facial_emotion, 0.3))
        voice_score = positive_emotions.get(voice_emotion, negative_emotions.get(voice_emotion, 0.3))
        
        # Factor in prosody (normalized pitch and intensity indicate positivity)
        prosody_factor = min(1.0, (prosody_scores[0] / 200.0 + prosody_scores[1] * 10) / 2)
        
        return (face_score * 0.4 + voice_score * 0.4 + prosody_factor * 0.2) * 100

    def calculate_engagement(self, prosody_scores, facial_emotion):
        # High intensity, varied pitch, appropriate tempo indicate engagement
        intensity_score = min(1.0, prosody_scores[1] * 10)
        tempo_score = min(1.0, prosody_scores[2] / 120.0)
        pitch_variation = min(1.0, prosody_scores[0] / 150.0)
        
        engagement_emotions = {'happy': 0.8, 'surprise': 0.9, 'angry': 0.7}
        emotion_factor = engagement_emotions.get(facial_emotion, 0.4)
        
        return (intensity_score * 0.3 + tempo_score * 0.3 + pitch_variation * 0.2 + emotion_factor * 0.2) * 100

    def calculate_confidence(self, prosody_scores, voice_emotion):
        # Steady pitch, good intensity, low pause ratio indicate confidence
        pitch_stability = max(0, 1.0 - (prosody_scores[0] / 300.0))  # Lower pitch variation = more confident
        intensity_score = min(1.0, prosody_scores[1] * 8)
        pause_factor = max(0, 1.0 - prosody_scores[3])  # Less pauses = more confident
        
        confident_emotions = {'happy': 0.8, 'neutral': 0.7, 'angry': 0.6}
        emotion_factor = confident_emotions.get(voice_emotion, 0.3)
        
        return (pitch_stability * 0.3 + intensity_score * 0.3 + pause_factor * 0.2 + emotion_factor * 0.2) * 100

    def calculate_neutrality(self, facial_emotion, voice_emotion):
        neutral_weight = {'neutral': 1.0, 'happy': 0.3, 'sad': 0.2, 'angry': 0.1, 
                         'fear': 0.1, 'surprise': 0.2, 'disgust': 0.1}
        
        face_neutral = neutral_weight.get(facial_emotion, 0.0)
        voice_neutral = neutral_weight.get(voice_emotion, 0.0)
        
        return (face_neutral + voice_neutral) * 50

    def generate_personalized_feedback(self, facial, voice, prosody, positivity, engagement, confidence, neutrality):
        feedback_parts = []
        
        # Emotional state assessment
        if positivity > 70:
            feedback_parts.append("You're displaying positive emotional energy! 🌟")
        elif positivity < 30:
            feedback_parts.append("You seem to be experiencing some challenging emotions. 💙")
        else:
            feedback_parts.append("Your emotional state appears balanced. ⚖️")
        
        # Engagement assessment
        if engagement > 70:
            feedback_parts.append("Your voice shows high engagement and enthusiasm! 🚀")
        elif engagement < 40:
            feedback_parts.append("Consider increasing vocal variety to enhance engagement. 📈")
        
        # Confidence assessment
        if confidence > 70:
            feedback_parts.append("You're projecting strong confidence through your speech patterns! 💪")
        elif confidence < 40:
            feedback_parts.append("Try speaking with more steady pace and volume for increased confidence. 🎯")
        
        # Specific prosody feedback
        pitch = prosody[0]
        intensity = prosody[1]
        tempo = prosody[2]
        
        if pitch > 200:
            feedback_parts.append("Your pitch is quite high - try lowering it slightly for a calmer effect. 🎵")
        elif pitch < 80:
            feedback_parts.append("Consider adding more pitch variation to keep listeners engaged. 🎶")
        
        if intensity < 0.02:
            feedback_parts.append("Try speaking with more volume and energy. 🔊")
        
        if tempo > 150:
            feedback_parts.append("You're speaking quite quickly - slowing down might improve clarity. 🐌")
        elif tempo < 60:
            feedback_parts.append("Consider increasing your speaking pace for better engagement. 🏃")
        
        return " ".join(feedback_parts)

# ========================================
# Real-time Multi-Modal Emotion Detector
# ========================================

class RealTimeEmotionDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.fer_model = None
        self.sound_model = None
        self.prosody_model = None
        self.feedback_generator = BARTFeedbackGenerator()
        
        # Load models
        self.load_models()
        
        # Audio setup
        self.chunk = 1024
        self.sample_rate = 16000
        self.audio_buffer = deque(maxlen=self.sample_rate * 3)  # 3 seconds buffer
        self.is_recording = False
        
        # Emotion labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Results storage
        self.latest_results = {
            'facial_emotion': 'neutral',
            'voice_emotion': 'neutral',
            'prosody_scores': np.zeros(5),
            'feedback': '',
            'scores': {}
        }
        
        # Threading
        self.running = False
        self.audio_thread = None
        self.processing_thread = None
        
    def test_models(self):
        """Test all models to ensure they're working"""
        print("Testing models...")
        
        # Test FER model with dummy data
        try:
            dummy_face = torch.randn(1, 1, 48, 48).to(self.device)
            with torch.no_grad():
                fer_output = self.fer_model(dummy_face)
                print(f"✓ FER model output shape: {fer_output.shape}")
        except Exception as e:
            print(f"✗ FER model test failed: {e}")
        
        # Test sound model with dummy data
        try:
            dummy_audio = torch.randn(1, 1, 40, 216).to(self.device)
            with torch.no_grad():
                sound_output = self.sound_model(dummy_audio)
                print(f"✓ Sound model output shape: {sound_output.shape}")
        except Exception as e:
            print(f"✗ Sound model test failed: {e}")
        
        # Test prosody model with dummy data
        try:
            dummy_mel = torch.randn(1, 1, 64, 128).to(self.device)
            with torch.no_grad():
                prosody_output = self.prosody_model(dummy_mel)
                print(f"✓ Prosody model output shape: {prosody_output.shape}")
        except Exception as e:
            print(f"✗ Prosody model test failed: {e}")
        
        # Test camera
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera working - frame shape: {frame.shape}")
            else:
                print("✗ Camera not accessible")
            cap.release()
        except Exception as e:
            print(f"✗ Camera test failed: {e}")
        
        print("Model testing complete!\n")

    def load_models(self):
        print("Loading models...")
        
        # Load FER CNN model
        try:
            self.fer_model = FER_CNN(num_classes=7).to(self.device)
            # You need to provide the path to your trained model
            self.fer_model.load_state_dict(torch.load('fer_model_finetuned_v2.pth', map_location=self.device))
            self.fer_model.eval()
            print("✓ FER CNN model loaded")
        except Exception as e:
            print(f"✗ Failed to load FER model: {e}")
            print("Creating random FER model for demo...")
            self.fer_model = FER_CNN(num_classes=7).to(self.device)
            self.fer_model.eval()
        
        # Load Sound Emotion model
        try:
            self.sound_model = TwoLayerCNN(input_height=40, input_width=216, num_classes=7).to(self.device)
            self.sound_model.load_state_dict(torch.load('sound_emotion_model.pth', map_location=self.device))
            self.sound_model.eval()
            print("✓ Sound emotion model loaded")
        except Exception as e:
            print(f"✗ Failed to load sound model: {e}")
            print("Creating random sound model for demo...")
            self.sound_model = TwoLayerCNN(input_height=40, input_width=216, num_classes=7).to(self.device)
            self.sound_model.eval()
        
        # Load Prosody model
        try:
            self.prosody_model = ProsodyNet(num_targets=5).to(self.device)
            checkpoint = torch.load('prosody_net_best.pth', map_location=self.device)
            self.prosody_model.load_state_dict(checkpoint['model_state_dict'])
            self.prosody_model.eval()
            print("✓ Prosody model loaded")
        except Exception as e:
            print(f"✗ Failed to load prosody model: {e}")
            print("Creating random prosody model for demo...")
            self.prosody_model = ProsodyNet(num_targets=5).to(self.device)
            self.prosody_model.eval()
        
        # Test all models
        self.test_models()

    def detect_and_draw_faces(self, frame):
        """Detect faces and return processed frame with bounding boxes and emotions"""
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize face cascade
        if not hasattr(self, 'face_cascade'):
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces with better parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        processed_frame = frame.copy()
        best_emotion = 'neutral'
        best_confidence = 0.0
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_gray = gray[y:y+h, x:x+w]
            
            # Preprocess face for emotion detection
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized.astype('float32') / 255.0
            
            # Apply proper normalization (mean=0.5, std=0.5 for grayscale)
            face_normalized = (face_normalized - 0.5) / 0.5
            
            # Convert to tensor
            face_tensor = torch.tensor(face_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Predict emotion
            with torch.no_grad():
                outputs = self.fer_model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion = self.emotion_labels[predicted.item()]
                conf = confidence.item()
                
                # Keep track of the most confident prediction
                if conf > best_confidence:
                    best_emotion = emotion
                    best_confidence = conf
            
            # Draw bounding box
            color = self.get_emotion_color(emotion)
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label with confidence
            label = f"{emotion.capitalize()}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw background for text
            cv2.rectangle(processed_frame, 
                         (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), 
                         color, -1)
            
            # Draw text
            cv2.putText(processed_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw additional info
            info_text = f"Face: {w}x{h}"
            cv2.putText(processed_frame, info_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return processed_frame, best_emotion, best_confidence, len(faces)
    
    def get_emotion_color(self, emotion):
        """Return BGR color for each emotion"""
        emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 100, 0),    # Dark Green
            'fear': (128, 0, 128),     # Purple
            'happy': (0, 255, 0),      # Green
            'neutral': (128, 128, 128), # Gray
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 255, 255)  # Yellow
        }
        return emotion_colors.get(emotion, (255, 255, 255))

    def preprocess_face(self, frame):
        """Legacy function for backward compatibility"""
        processed_frame, emotion, confidence, face_count = self.detect_and_draw_faces(frame)
        return processed_frame if face_count > 0 else None

    def predict_facial_emotion(self, frame):
        """Enhanced facial emotion prediction with better preprocessing"""
        processed_frame, emotion, confidence, face_count = self.detect_and_draw_faces(frame)
        
        if face_count == 0:
            return 'neutral', 0.0, frame
        
        return emotion, confidence, processed_frame

    def predict_voice_emotion(self, audio_data):
        if len(audio_data) < self.sample_rate:  # Need at least 1 second
            return 'neutral', 0.0
        
        # Extract MFCC features
        mfcc = extract_mfcc_features(audio_data, sr=self.sample_rate)
        mfcc_tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.sound_model(mfcc_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return self.emotion_labels[predicted.item()], confidence.item()

    def predict_prosody(self, audio_data):
        if len(audio_data) < self.sample_rate:
            return np.zeros(5, dtype=np.float32)
        
        # Extract prosody features directly
        prosody_features = extract_prosody_features(audio_data, sr=self.sample_rate)
        
        # Also run through prosody network for completeness
        mel_spec = extract_mel_spectrogram(audio_data, sr=self.sample_rate)
        mel_tensor = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prosody_pred = self.prosody_model(mel_tensor)
            prosody_pred = prosody_pred.cpu().numpy().flatten()
        
        # Return the direct features (more interpretable)
        return prosody_features

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)
        return (in_data, pyaudio.paContinue)

    def start_audio_stream(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        self.is_recording = True

    def stop_audio_stream(self):
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        self.is_recording = False

    def process_multimodal_emotions(self):
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                continue
            
            frame_count += 1
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get current audio data
            audio_data = np.array(list(self.audio_buffer))
            
            # Predict emotions (process every 3rd frame for performance)
            if frame_count % 3 == 0:
                facial_emotion, face_conf, processed_frame = self.predict_facial_emotion(frame)
                voice_emotion, voice_conf = self.predict_voice_emotion(audio_data)
                prosody_scores = self.predict_prosody(audio_data)
                
                # Add performance info to frame
                self.add_debug_info(processed_frame, facial_emotion, voice_emotion, prosody_scores, audio_data)
                
                # Generate feedback (less frequently to avoid spam)
                if frame_count % 30 == 0:  # Every 30 frames (~3 seconds at 10fps)
                    try:
                        feedback_result = self.feedback_generator.generate_feedback(
                            facial_emotion, voice_emotion, prosody_scores
                        )
                        feedback_text = feedback_result['feedback']
                        scores = feedback_result
                    except Exception as e:
                        print(f"Feedback generation error: {e}")
                        feedback_text = f"Facial: {facial_emotion}, Voice: {voice_emotion}"
                        scores = {
                            'positivity': 50, 'engagement': 50, 
                            'confidence': 50, 'neutrality': 50
                        }
                else:
                    # Use previous feedback
                    feedback_text = self.latest_results.get('feedback', '')
                    scores = self.latest_results.get('scores', {
                        'positivity': 50, 'engagement': 50, 
                        'confidence': 50, 'neutrality': 50
                    })
                
                # Update results
                self.latest_results = {
                    'facial_emotion': facial_emotion,
                    'voice_emotion': voice_emotion,
                    'prosody_scores': prosody_scores,
                    'feedback': feedback_text,
                    'scores': scores,
                    'face_confidence': face_conf,
                    'voice_confidence': voice_conf,
                    'frame': processed_frame,
                    'audio_length': len(audio_data),
                    'frame_count': frame_count
                }
            else:
                # Just update the frame without processing
                self.latest_results['frame'] = frame
            
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()

    def add_debug_info(self, frame, facial_emotion, voice_emotion, prosody_scores, audio_data):
        """Add debugging information to the frame"""
        h, w = frame.shape[:2]
        
        # Add status information
        status_info = [
            f"Facial: {facial_emotion.capitalize()}",
            f"Voice: {voice_emotion.capitalize()}",
            f"Audio Length: {len(audio_data)} samples",
            f"Pitch: {prosody_scores[0]:.1f}",
            f"Intensity: {prosody_scores[1]:.3f}",
            f"Tempo: {prosody_scores[2]:.1f}"
        ]
        
        # Draw background for status
        cv2.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 150), (255, 255, 255), 2)
        
        # Draw status text
        for i, info in enumerate(status_info):
            y_pos = 30 + i * 20
            cv2.putText(frame, info, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add crosshair for face positioning
        cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 1)
        cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 1)

    def start_detection(self):
        self.running = True
        self.start_audio_stream()
        self.processing_thread = threading.Thread(target=self.process_multimodal_emotions)
        self.processing_thread.start()

    def stop_detection(self):
        self.running = False
        self.stop_audio_stream()
        if self.processing_thread:
            self.processing_thread.join()

# ========================================
# GUI Application
# ========================================

class EmotionDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Multi-Modal Emotion Detection System")
        self.root.geometry("1400x900")
        
        # Initialize detector
        self.detector = RealTimeEmotionDetector()
        
        # Create GUI elements
        self.create_widgets()
        
        # Start update loop
        self.update_display()

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for video and real-time info
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video frame
        video_frame = ttk.LabelFrame(top_frame, text="Live Video Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = tk.Label(video_frame)
        self.video_label.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(top_frame, text="Emotion Analysis")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Emotion displays
        ttk.Label(results_frame, text="Facial Emotion:", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(10,5))
        self.facial_emotion_var = tk.StringVar(value="neutral")
        self.facial_label = ttk.Label(results_frame, textvariable=self.facial_emotion_var, font=('Arial', 16))
        self.facial_label.pack(anchor='w', padx=20)
        
        ttk.Label(results_frame, text="Voice Emotion:", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(10,5))
        self.voice_emotion_var = tk.StringVar(value="neutral")
        self.voice_label = ttk.Label(results_frame, textvariable=self.voice_emotion_var, font=('Arial', 16))
        self.voice_label.pack(anchor='w', padx=20)
        
        # Scores frame
        scores_frame = ttk.LabelFrame(results_frame, text="Emotion Scores")
        scores_frame.pack(fill=tk.BOTH, expand=True, pady=(10,0))
        
        self.score_vars = {}
        score_names = ['Positivity', 'Engagement', 'Confidence', 'Neutrality']
        
        for score_name in score_names:
            frame = ttk.Frame(scores_frame)
            frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(frame, text=f"{score_name}:", width=12).pack(side=tk.LEFT)
            
            self.score_vars[score_name.lower()] = tk.StringVar(value="0%")
            score_label = ttk.Label(frame, textvariable=self.score_vars[score_name.lower()], font=('Arial', 10, 'bold'))
            score_label.pack(side=tk.LEFT, padx=(10, 0))
            
            # Progress bar
            progress = ttk.Progressbar(frame, length=150, mode='determinate')
            progress.pack(side=tk.RIGHT, padx=(10, 0))
            self.score_vars[f'{score_name.lower()}_progress'] = progress
        
        # Prosody info
        prosody_frame = ttk.LabelFrame(results_frame, text="Voice Prosody")
        prosody_frame.pack(fill=tk.X, pady=(10,0))
        
        prosody_labels = ['Pitch', 'Intensity', 'Tempo', 'Pause Ratio', 'Stress']
        self.prosody_vars = {}
        
        for label in prosody_labels:
            frame = ttk.Frame(prosody_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=f"{label}:", width=12).pack(side=tk.LEFT)
            self.prosody_vars[label.lower().replace(' ', '_')] = tk.StringVar(value="0.00")
            ttk.Label(frame, textvariable=self.prosody_vars[label.lower().replace(' ', '_')], font=('Arial', 9)).pack(side=tk.LEFT)
        
        # Bottom frame for feedback
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Feedback display
        feedback_frame = ttk.LabelFrame(bottom_frame, text="AI Feedback & Recommendations")
        feedback_frame.pack(fill=tk.BOTH, expand=True)
        
        self.feedback_text = scrolledtext.ScrolledText(feedback_frame, height=8, wrap=tk.WORD, font=('Arial', 11))
        self.feedback_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(control_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Detection", command=self.stop_detection, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(control_frame, text="Clear Feedback", command=self.clear_feedback)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to start detection")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT)

    def start_detection(self):
        self.detector.start_detection()
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_var.set("Detection running...")

    def stop_detection(self):
        self.detector.stop_detection()
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_var.set("Detection stopped")

    def clear_feedback(self):
        self.feedback_text.delete(1.0, tk.END)

    def update_display(self):
        try:
            results = self.detector.latest_results
            
            # Update emotion labels
            self.facial_emotion_var.set(f"{results['facial_emotion'].title()} ({results.get('face_confidence', 0):.2f})")
            self.voice_emotion_var.set(f"{results['voice_emotion'].title()} ({results.get('voice_confidence', 0):.2f})")
            
            # Update scores
            if 'scores' in results and results['scores']:
                for score_name, value in results['scores'].items():
                    if score_name in self.score_vars:
                        self.score_vars[score_name].set(f"{value:.1f}%")
                        if f'{score_name}_progress' in self.score_vars:
                            self.score_vars[f'{score_name}_progress']['value'] = value
            
            # Update prosody information
            prosody = results['prosody_scores']
            prosody_labels = ['pitch', 'intensity', 'tempo', 'pause_ratio', 'stress']
            for i, label in enumerate(prosody_labels):
                if label in self.prosody_vars and i < len(prosody):
                    self.prosody_vars[label].set(f"{prosody[i]:.2f}")
            
            # Update video feed
            if 'frame' in results:
                frame = results['frame']
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (400, 300))
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image)
                self.video_label.config(image=photo)
                self.video_label.image = photo
            
            # Update feedback (append new feedback)
            if results['feedback'] and results['feedback'] != getattr(self, 'last_feedback', ''):
                self.feedback_text.insert(tk.END, f"\n[{time.strftime('%H:%M:%S')}] {results['feedback']}\n")
                self.feedback_text.see(tk.END)
                self.last_feedback = results['feedback']
                
        except Exception as e:
            print(f"Display update error: {e}")
        
        # Schedule next update
        self.root.after(100, self.update_display)

    def on_closing(self):
        self.detector.stop_detection()
        self.root.destroy()

# ========================================
# Model Creation and Training Functions (for when models don't exist)
# ========================================

def create_dummy_models():
    """Create and save dummy models for demonstration when trained models are not available"""
    print("Creating dummy models for demonstration...")
    
    # Create dummy FER model
    fer_model = FER_CNN(num_classes=7)
    torch.save(fer_model.state_dict(), 'fer_model_finetuned_v2.pth')
    print("✓ Created dummy FER model")
    
    # Create dummy sound emotion model
    sound_model = TwoLayerCNN(input_height=40, input_width=216, num_classes=7)
    torch.save(sound_model.state_dict(), 'sound_emotion_model.pth')
    print("✓ Created dummy sound emotion model")
    
    # Create dummy prosody model with normalization stats
    prosody_model = ProsodyNet(num_targets=5)
    dummy_stats = {
        'model_state_dict': prosody_model.state_dict(),
        'mean': np.array([120.0, 0.05, 90.0, 0.2, 0.1]),
        'std': np.array([50.0, 0.02, 30.0, 0.15, 0.05])
    }
    torch.save(dummy_stats, 'prosody_net_best.pth')
    print("✓ Created dummy prosody model")

# ========================================
# Enhanced Visualization Functions
# ========================================

class EmotionVisualizer:
    def __init__(self, detector):
        self.detector = detector
        self.emotion_history = deque(maxlen=100)
        self.score_history = {
            'positivity': deque(maxlen=100),
            'engagement': deque(maxlen=100),
            'confidence': deque(maxlen=100),
            'neutrality': deque(maxlen=100)
        }
        
    def update_history(self, results):
        self.emotion_history.append({
            'facial': results['facial_emotion'],
            'voice': results['voice_emotion'],
            'timestamp': time.time()
        })
        
        if 'scores' in results:
            for score_name, value in results['scores'].items():
                if score_name in self.score_history:
                    self.score_history[score_name].append(value)

    def create_emotion_plot(self):
        """Create a matplotlib plot of emotion trends over time"""
        if not self.emotion_history:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot emotion distribution
        emotions = [entry['facial'] for entry in self.emotion_history]
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
        
        ax1.bar(emotion_counts.keys(), emotion_counts.values())
        ax1.set_title('Facial Emotion Distribution (Last 100 frames)')
        ax1.set_ylabel('Frequency')
        
        # Plot score trends
        for score_name, history in self.score_history.items():
            if history:
                ax2.plot(list(history), label=score_name.title())
        
        ax2.set_title('Emotion Scores Over Time')
        ax2.set_ylabel('Score (%)')
        ax2.set_xlabel('Time Steps')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# ========================================
# Configuration and Settings
# ========================================

class EmotionDetectorConfig:
    def __init__(self):
        self.MODEL_PATHS = {
            'fer': 'fer_model_finetuned_v2.pth',
            'sound': 'sound_emotion_model.pth',
            'prosody': 'prosody_net_best.pth'
        }
        
        self.AUDIO_CONFIG = {
            'sample_rate': 16000,
            'chunk_size': 1024,
            'buffer_seconds': 3
        }
        
        self.VIDEO_CONFIG = {
            'fps': 10,
            'frame_width': 640,
            'frame_height': 480
        }
        
        self.FEEDBACK_CONFIG = {
            'update_interval': 2.0,  # seconds
            'history_length': 50
        }

# ========================================
# Utility Functions
# ========================================

def check_dependencies():
    """Check if all required dependencies are available"""
    # Map pip package names to their import names
    package_mapping = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'transformers': 'transformers',
        'librosa': 'librosa',
        'pyaudio': 'pyaudio',
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'Pillow': 'PIL',
        'tkinter': 'tkinter'
    }
    
    missing_packages = []
    for pip_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("Missing packages:", missing_packages)
        print("Install them using: pip install", ' '.join(missing_packages))
        return False
    return True

def setup_logging():
    """Setup logging for the application"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('emotion_detector.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ========================================
# Performance Monitoring
# ========================================

class PerformanceMonitor:
    def __init__(self):
        self.processing_times = deque(maxlen=100)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def start_timer(self):
        self.start_time = time.time()
        
    def end_timer(self):
        processing_time = time.time() - self.start_time
        self.processing_times.append(processing_time)
        self.fps_counter += 1
        
    def get_average_processing_time(self):
        return np.mean(self.processing_times) if self.processing_times else 0
        
    def get_fps(self):
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
            return fps
        return 0

# ========================================
# Main Application Entry Point
# ========================================

def main():
    """Main function to run the emotion detection system"""
    
    # Check dependencies (with option to skip)
    try:
        if not check_dependencies():
            print("Some dependencies might be missing, but attempting to run anyway...")
            print("If you encounter import errors, please install the missing packages.")
    except Exception as e:
        print(f"Dependency check failed: {e}")
        print("Proceeding with application startup...")
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Real-time Multi-Modal Emotion Detection System")
    
    try:
        # Create dummy models if they don't exist (for demo purposes)
        import os
        model_files = ['fer_model_finetuned_v2.pth', 'sound_emotion_model.pth', 'prosody_net_best.pth']
        if not all(os.path.exists(f) for f in model_files):
            print("Trained models not found. Creating dummy models for demonstration...")
            create_dummy_models()
        
        # Create and run GUI application
        root = tk.Tk()
        app = EmotionDetectorGUI(root)
        
        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        logger.info("GUI initialized successfully")
        print("\n" + "="*60)
        print("REAL-TIME MULTI-MODAL EMOTION DETECTION SYSTEM")
        print("="*60)
        print("Features:")
        print("• Real-time facial emotion recognition")
        print("• Voice emotion analysis")
        print("• Prosody feature extraction (pitch, intensity, tempo)")
        print("• AI-powered feedback using Facebook BART-large-mnli")
        print("• Multi-modal emotion scoring")
        print("\nInstructions:")
        print("1. Click 'Start Detection' to begin")
        print("2. Position yourself in front of the camera")
        print("3. Speak naturally for voice analysis")
        print("4. View real-time results and AI feedback")
        print("5. Click 'Stop Detection' when done")
        print("="*60)
        
        # Start the GUI main loop
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    main()
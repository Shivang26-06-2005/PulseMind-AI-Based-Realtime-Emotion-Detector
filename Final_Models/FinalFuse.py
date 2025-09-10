import cv2
import torch
import numpy as np
import librosa
import pyaudio
import threading
import time
from collections import deque
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
from scipy.signal import find_peaks
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# ============================
# 1. Device Setup
# ============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================
# 2. Video Emotion Model (FER_CNN)
# ============================
class FER_CNN(nn.Module):
    def __init__(self, num_classes):
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
            nn.Linear(128 * 6 * 6, 256),
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

# ============================
# 3. Sound Emotion Model (CNN)
# ============================
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

# ============================
# 4. Prosody Model (CNN+LSTM)
# ============================
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

# ============================
# 5. Feature Extraction Functions
# ============================
def extract_prosody_features(audio_data, sr=16000):
    """Extract prosody features from audio data"""
    y = audio_data
    duration = len(y) / sr
    
    # Pitch
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_vals = [pitches[mags[:, i].argmax(), i] for i in range(pitches.shape[1])]
    pitch_vals = np.array([p for p in pitch_vals if p > 0])
    pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
    
    # Intensity (RMS)
    rms = librosa.feature.rms(y=y)[0]
    intensity = float(np.mean(rms))
    
    # Tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
    except:
        tempo = 0.0
    
    # Pause Ratio
    intervals = librosa.effects.split(y, top_db=30)
    pauses = []
    prev_end = 0
    for start, end in intervals:
        pauses.append((prev_end, start))
        prev_end = end
    pause_ratio = float(sum([e - s for s, e in pauses]) / duration) if duration > 0 else 0.0
    
    # Stress Pattern
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
    """Extract MFCC features for sound emotion detection"""
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def extract_mel_spectrogram(audio_data, sr=16000, max_len=128):
    """Extract mel-spectrogram for prosody model"""
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0,0),(0,max_len - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
    return mel_db

# ============================
# 6. Multi-Modal Emotion Detector Class
# ============================
class MultiModalEmotionDetector:
    def __init__(self):
        # Emotion labels
        self.video_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.sound_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Load models
        self.load_models()
        
        # Initialize BART for feedback
        self.init_bart_feedback()
        
        # Video capture and face detection
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Audio setup
        self.setup_audio()
        
        # Transformation for video
        self.video_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Detection intervals and buffers
        self.face_locations = []
        self.face_labels = []
        self.last_video_detection = time.time()
        self.last_audio_detection = time.time()
        self.last_feedback_analysis = time.time()
        
        # Different intervals for different tasks
        self.video_detection_interval = 0.3  # Video emotion every 300ms
        self.audio_detection_interval = 1.0  # Audio emotion every 1s
        self.feedback_analysis_interval = 2.0  # BART feedback every 2s
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(16000 * 2))  # Reduced to 2 seconds
        self.audio_lock = threading.Lock()
        
        # Processing flags and threading
        self.processing_audio = False
        self.processing_video = False
        self.processing_feedback = False
        
        # Results storage with thread locks
        self.results_lock = threading.Lock()
        self.current_results = {
            'video_emotion': 'Neutral',
            'sound_emotion': 'neutral',
            'prosody_features': [0.0] * 5,
            'feedback_scores': {
                'engagement': 0.5,
                'confidence': 0.5,
                'nervousness': 0.5,
                'positivity': 0.5
            }
        }
        
        # Frame processing optimization
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
        
    def load_models(self):
        """Load all pre-trained models"""
        try:
            # Video emotion model
            self.video_model = FER_CNN(7).to(DEVICE)
            self.video_model.load_state_dict(torch.load('fer_model_finetuned_v2.pth', map_location=DEVICE))
            self.video_model.eval()
            print("✓ Video emotion model loaded")
        except FileNotFoundError:
            print("⚠ Video model not found, creating dummy model")
            self.video_model = FER_CNN(7).to(DEVICE)
            self.video_model.eval()
        
        try:
            # Sound emotion model - determine correct dimensions from checkpoint
            checkpoint = torch.load('sound_emotion_model.pth', map_location=DEVICE)
            
            # Extract the expected input size from the checkpoint
            fc1_weight_shape = checkpoint['fc1.weight'].shape
            expected_input_size = fc1_weight_shape[1]  # This is 7360
            
            # Calculate the corresponding height and width
            # We know it's 32 * (height//4) * (width//4) = expected_input_size
            # So (height//4) * (width//4) = expected_input_size / 32
            feature_map_size = expected_input_size // 32  # This should be 230
            
            # For MFCC with 40 features, if height=40, then width//4 = 230/10 = 23
            # So original width = 23*4 = 92
            input_height, input_width = 40, 92
            
            self.sound_model = TwoLayerCNN(input_height, input_width, 7).to(DEVICE)
            self.sound_model.load_state_dict(checkpoint)
            self.sound_model.eval()
            print(f"✓ Sound emotion model loaded with dimensions {input_height}x{input_width}")
        except FileNotFoundError:
            print("⚠ Sound model not found, creating dummy model")
            self.sound_model = TwoLayerCNN(40, 92, 7).to(DEVICE)
            self.sound_model.eval()
        except Exception as e:
            print(f"⚠ Sound model loading failed: {e}")
            # Fallback to default dimensions
            self.sound_model = TwoLayerCNN(40, 92, 7).to(DEVICE)
            self.sound_model.eval()
            
        try:
            # Prosody model
            checkpoint = torch.load('prosody_net_best.pth', map_location=DEVICE)
            self.prosody_model = ProsodyNet().to(DEVICE)
            self.prosody_model.load_state_dict(checkpoint['model_state_dict'])
            self.prosody_mean = checkpoint['mean']
            self.prosody_std = checkpoint['std']
            self.prosody_model.eval()
            print("✓ Prosody model loaded")
        except FileNotFoundError:
            print("⚠ Prosody model not found, creating dummy model")
            self.prosody_model = ProsodyNet().to(DEVICE)
            self.prosody_mean = np.zeros(5)
            self.prosody_std = np.ones(5)
            self.prosody_model.eval()
    
    def init_bart_feedback(self):
        """Initialize BART model for feedback analysis"""
        try:
            self.bart_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
            self.bart_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
            print("✓ BART feedback model loaded")
        except Exception as e:
            print(f"⚠ BART model failed to load: {e}")
            self.bart_tokenizer = None
            self.bart_model = None
    
    def setup_audio(self):
        """Setup audio recording"""
        self.audio = pyaudio.PyAudio()
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        
        try:
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            print("✓ Audio stream started")
        except Exception as e:
            print(f"⚠ Audio setup failed: {e}")
            self.stream = None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        with self.audio_lock:
            self.audio_buffer.extend(audio_data)
        return (None, pyaudio.paContinue)
    
    def analyze_with_bart(self, emotion_text):
        """Analyze emotions using BART for feedback scores"""
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
                # Create hypothesis for classification
                hypothesis = f"This shows high {aspect}"
                
                inputs = self.bart_tokenizer(text, hypothesis, return_tensors='pt', truncation=True)
                with torch.no_grad():
                    outputs = self.bart_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    # Get probability of entailment (index 2)
                    score = probs[0][2].item()
                    scores[aspect] = score
            except Exception as e:
                print(f"BART analysis failed for {aspect}: {e}")
                scores[aspect] = 0.5
                
        return scores
    
    def detect_video_emotion_threaded(self, frame):
        """Detect emotion from video frame in separate thread"""
        if self.processing_video:
            return  # Skip if already processing
        
        self.processing_video = True
        
        def process():
            try:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (320, 240))
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # Scale factor for face detection (faster with larger scale factor)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Scale back coordinates to original frame size
                    scale_x = frame.shape[1] / 320
                    scale_y = frame.shape[0] / 240
                    scaled_faces = [(int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)) 
                                   for (x, y, w, h) in faces]
                    
                    self.face_locations = scaled_faces
                    
                    # Process first face only
                    x, y, w, h = scaled_faces[0]
                    face_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
                    
                    # Resize face for model (faster processing)
                    face_roi = cv2.resize(face_roi, (48, 48))
                    face_pil = Image.fromarray(face_roi)
                    face_tensor = self.video_transform(face_pil).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        output = self.video_model(face_tensor)
                        pred = torch.argmax(output, 1).item()
                        emotion = self.video_emotions[pred]
                        
                        with self.results_lock:
                            self.current_results['video_emotion'] = emotion
                            self.face_labels = [emotion]
                else:
                    self.face_locations = []
                    self.face_labels = []
                    
            except Exception as e:
                print(f"Video processing error: {e}")
            finally:
                self.processing_video = False
        
        # Run in thread for non-blocking processing
        threading.Thread(target=process, daemon=True).start()
    
    def detect_sound_emotion_threaded(self):
        """Detect emotion from audio buffer in separate thread"""
        if self.processing_audio:
            return
        
        self.processing_audio = True
        
        def process():
            try:
                with self.audio_lock:
                    if len(self.audio_buffer) < 8000:  # Reduced minimum requirement
                        return
                    audio_data = np.array(list(self.audio_buffer))
                
                # Extract MFCC features
                mfcc = extract_mfcc_features(audio_data)
                mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = self.sound_model(mfcc_tensor)
                    pred = torch.argmax(output, 1).item()
                    emotion = self.sound_emotions[pred]
                    
                    with self.results_lock:
                        self.current_results['sound_emotion'] = emotion
                        
            except Exception as e:
                print(f"Sound processing error: {e}")
            finally:
                self.processing_audio = False
        
        threading.Thread(target=process, daemon=True).start()
    
    def detect_prosody_features_threaded(self):
        """Detect prosody features in separate thread"""
        def process():
            try:
                with self.audio_lock:
                    if len(self.audio_buffer) < 8000:
                        return
                    audio_data = np.array(list(self.audio_buffer))
                
                # Extract prosody features (lightweight)
                prosody_feats = extract_prosody_features(audio_data)
                
                with self.results_lock:
                    self.current_results['prosody_features'] = prosody_feats.tolist()
                    
            except Exception as e:
                print(f"Prosody processing error: {e}")
        
        threading.Thread(target=process, daemon=True).start()
    
    def analyze_feedback_threaded(self):
        """Analyze with BART in separate thread"""
        if self.processing_feedback:
            return
            
        self.processing_feedback = True
        
        def process():
            try:
                with self.results_lock:
                    video_emotion = self.current_results['video_emotion']
                    sound_emotion = self.current_results['sound_emotion']
                
                combined_emotion_text = f"{video_emotion} and {sound_emotion}"
                feedback_scores = self.analyze_with_bart(combined_emotion_text)
                
                with self.results_lock:
                    self.current_results['feedback_scores'] = feedback_scores
                    
            except Exception as e:
                print(f"Feedback processing error: {e}")
            finally:
                self.processing_feedback = False
        
        threading.Thread(target=process, daemon=True).start()
    
    def draw_results(self, frame):
        """Draw all results on the frame (optimized)"""
        height, width = frame.shape[:2]
        
        # Draw face rectangles and emotions (only if we have recent detections)
        for i, (x, y, w, h) in enumerate(self.face_locations):
            label = self.face_labels[i] if i < len(self.face_labels) else ''
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Video: {label}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Create smaller information panel for better performance
        panel_height = 200
        panel_width = min(400, width)  # Limit panel width
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Get current results safely
        with self.results_lock:
            video_emotion = self.current_results['video_emotion']
            sound_emotion = self.current_results['sound_emotion']
            prosody_feats = self.current_results['prosody_features']
            feedback_scores = self.current_results['feedback_scores']
        
        # Optimized text rendering
        y_pos = 25
        line_height = 20
        font_scale = 0.5
        thickness = 1
        
        # Emotion results
        cv2.putText(panel, f"Video: {video_emotion}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        y_pos += line_height
        
        cv2.putText(panel, f"Sound: {sound_emotion}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        y_pos += line_height
        
        # Prosody features (simplified display)
        prosody_summary = f"Pitch:{prosody_feats[0]:.1f} Int:{prosody_feats[1]:.2f}"
        cv2.putText(panel, f"Prosody: {prosody_summary}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
        y_pos += line_height
        
        # Feedback scores (compact display)
        cv2.putText(panel, "Feedback Scores:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
        y_pos += line_height
        
        for aspect, score in feedback_scores.items():
            color = (0, 255, 0) if score > 0.6 else (0, 255, 255) if score > 0.4 else (0, 0, 255)
            cv2.putText(panel, f"{aspect[:3]}: {score:.2f}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            y_pos += line_height
        
        # Resize panel to match frame width if needed
        if panel_width < width:
            panel_resized = np.zeros((panel_height, width, 3), dtype=np.uint8)
            panel_resized[:, :panel_width] = panel
            panel = panel_resized
        
        # Combine frame and panel
        combined = np.vstack([frame, panel])
        return combined
    
    def run(self):
        """Main detection loop (optimized)"""
        print("Starting Multi-Modal Emotion Detection...")
        print("Press 'q' to quit")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = time.time()
            self.frame_count += 1
            
            # Skip frames for performance
            if self.frame_count % self.frame_skip != 0:
                display_frame = self.draw_results(frame)
                cv2.imshow('Multi-Modal Emotion Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Video emotion detection (most frequent)
            if current_time - self.last_video_detection >= self.video_detection_interval:
                self.detect_video_emotion_threaded(frame)
                self.last_video_detection = current_time
            
            # Audio emotion detection (less frequent)
            if current_time - self.last_audio_detection >= self.audio_detection_interval:
                self.detect_sound_emotion_threaded()
                self.detect_prosody_features_threaded()
                self.last_audio_detection = current_time
            
            # BART feedback analysis (least frequent, most expensive)
            if current_time - self.last_feedback_analysis >= self.feedback_analysis_interval:
                self.analyze_feedback_threaded()
                self.last_feedback_analysis = current_time
            
            # Draw results on frame
            display_frame = self.draw_results(frame)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps_elapsed = time.time() - fps_start_time
                fps = fps_counter / fps_elapsed
                fps_counter = 0
                fps_start_time = time.time()
                
                # Print periodic updates instead of every frame
                with self.results_lock:
                    results_copy = self.current_results.copy()
                
                print(f"\n=== FPS: {fps:.1f} ===")
                print(f"Video: {results_copy['video_emotion']}, Sound: {results_copy['sound_emotion']}")
                print(f"Prosody: {[f'{x:.1f}' for x in results_copy['prosody_features'][:3]]}")
                print(f"Feedback: {[(k[:3], f'{v:.2f}') for k, v in results_copy['feedback_scores'].items()]}")
            
            # Show frame with minimal delay
            cv2.imshow('Multi-Modal Emotion Detection', display_frame)
            
            # Non-blocking key check
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        cv2.destroyAllWindows()
        print("Cleanup completed")

# ============================
# 7. Main Execution
# ============================
if __name__ == "__main__":
    try:
        detector = MultiModalEmotionDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
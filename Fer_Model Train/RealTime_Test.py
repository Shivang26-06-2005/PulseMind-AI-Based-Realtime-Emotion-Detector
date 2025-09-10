import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
import time

# -----------------------------
# 1. Device setup
# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# -----------------------------
# 2. Load your fine-tuned FER CNN
# -----------------------------
# Re-define the model architecture from the provided file
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

# Load pretrained fine-tuned weights
NUM_CLASSES = 7
model = FER_CNN(NUM_CLASSES).to(DEVICE)
try:
    # Make sure you have the 'fer_model_finetuned.pth' file in the same directory
    model.load_state_dict(torch.load('fer_model_finetuned_v2.pth', map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model weights file 'fer_model_finetuned.pth' not found. Please ensure it's in the same directory.")
    exit()

# -----------------------------
# 3. Transformation
# -----------------------------
# Define the transformations required for the model input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -----------------------------
# 4. OpenCV Video & Face Detection
# -----------------------------
# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)
# Initialize the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------
# 5. Real-time emotion detection loop
# -----------------------------
# We'll detect faces less frequently to improve performance
face_locations = []
face_labels = []
last_detection_time = time.time()
DETECTION_INTERVAL = 0.5  # seconds

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    current_time = time.time()
    # Only perform face detection and prediction at a regular interval
    if current_time - last_detection_time >= DETECTION_INTERVAL:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        face_locations = faces
        face_labels = [] # Reset labels
        
        if len(faces) > 0:
            # For each detected face, crop the region of interest (ROI)
            # and convert it to a PIL Image for model input
            rois = [Image.fromarray(gray[y:y+h, x:x+w]) for (x, y, w, h) in faces]
            # Transform the ROIs into PyTorch tensors
            tensors = torch.stack([transform(roi) for roi in rois]).to(DEVICE)
            
            with torch.no_grad():
                # Get the model's predictions
                outputs = model(tensors)
                # Get the index of the highest probability
                preds = torch.argmax(outputs, 1)
                # Map the prediction index to an emotion label
                face_labels = [EMOTIONS[p.item()] for p in preds]

        last_detection_time = current_time

    # Draw rectangles and labels on the original color frame
    for i, (x, y, w, h) in enumerate(face_locations):
        # Get the label for the current face
        label = face_labels[i] if i < len(face_labels) else ''
        
        # Draw the rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw the emotion label above the rectangle
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the final output frame
    cv2.imshow('Real-time Emotion Detector', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
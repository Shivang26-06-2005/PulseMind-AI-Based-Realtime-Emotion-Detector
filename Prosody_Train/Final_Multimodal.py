import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# 1. Load pretrained models
# =========================
from Sound_Model2 import TwoLayerCNN   # your emotion model definition
from Finetune_SoundTRainn import ProsodyNet    # your prosody model definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained emotion model
emotion_model = TwoLayerCNN(
    input_height=40,    # MFCC height
    input_width=200,    # example, use your max_len
    num_classes=7       # angry, disgust, fear, happy, neutral, sad, surprise
).to(device)
emotion_model.load_state_dict(torch.load("sound_emotion_model.pth"))

# Load pretrained prosody model
prosody_model = ProsodyNet(num_targets=5).to(device)
prosody_model.load_state_dict(torch.load("prosody_net_finetuned_all.pth"))

# =========================
# 2. Fusion Model
# =========================
class MultimodalNet(nn.Module):
    def __init__(self, emotion_model, prosody_model, num_classes):
        super().__init__()
        self.emotion_model = emotion_model
        self.prosody_model = prosody_model

        # Fusion layers
        self.fc_fusion = nn.Linear(128 + 128, 256)  # adjust sizes as per model outputs
        self.fc_out = nn.Linear(256, num_classes)

        # Uncertainty-based learnable weights
        self.log_sigma_emotion = nn.Parameter(torch.zeros(1))
        self.log_sigma_prosody = nn.Parameter(torch.zeros(1))

    def forward(self, x_emotion, x_prosody):
        # Get embeddings from each model
        feat_emotion = self.emotion_model.forward_features(x_emotion)  # shape (B,128)
        feat_prosody = self.prosody_model.forward_features(x_prosody)  # shape (B,128)

        # Fuse
        fused = torch.cat([feat_emotion, feat_prosody], dim=1)
        fused = torch.relu(self.fc_fusion(fused))
        out = self.fc_out(fused)
        return out, feat_emotion, feat_prosody

    def loss(self, pred, target_emotion, feat_prosody, target_prosody):
        # Classification loss (CrossEntropy)
        ce_loss = nn.CrossEntropyLoss()(pred, target_emotion)

        # Regression loss (MSE)
        mse_loss = nn.MSELoss()(feat_prosody, target_prosody)

        # Uncertainty-weighted total loss
        loss = (torch.exp(-self.log_sigma_emotion) * ce_loss +
                self.log_sigma_emotion +
                torch.exp(-self.log_sigma_prosody) * mse_loss +
                self.log_sigma_prosody)

        return loss, ce_loss.item(), mse_loss.item()


# Add helper methods in pretrained models to extract embeddings
def add_forward_features_emotion(model):
    def forward_features(x):
        x = model.pool(model.relu(model.bn1(model.conv1(x))))
        x = model.pool(model.relu(model.bn2(model.conv2(x))))
        x = x.view(x.size(0), -1)
        x = model.relu(model.fc1(x))  # embedding before dropout & fc2
        return x
    model.forward_features = forward_features
add_forward_features_emotion(emotion_model)

def add_forward_features_prosody(model):
    def forward_features(x):
        x = model.pool(torch.relu(model.bn1(model.conv1(x))))
        x = model.pool(torch.relu(model.bn2(model.conv2(x))))
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C*H)
        out, _ = model.lstm(x)
        out = out[:, -1, :]  # last hidden state
        return out
    model.forward_features = forward_features
add_forward_features_prosody(prosody_model)

# =========================
# 3. Training Loop
# =========================
num_classes = 7
model = MultimodalNet(emotion_model, prosody_model, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    total_loss, total_ce, total_mse = 0, 0, 0
    for (X_emotion, y_emotion), (X_prosody, y_prosody) in zip(train_loader_emotion, train_loader_prosody):
        X_emotion, y_emotion = X_emotion.to(device), y_emotion.to(device)
        X_prosody, y_prosody = X_prosody.to(device), y_prosody.to(device)

        optimizer.zero_grad()
        pred, feat_emotion, feat_prosody = model(X_emotion, X_prosody)
        loss, ce_loss, mse_loss = model.loss(pred, y_emotion, feat_prosody, y_prosody)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss
        total_mse += mse_loss

    print(f"Epoch {epoch+1}, Loss={total_loss/len(train_loader_emotion):.4f}, "
          f"CE={total_ce/len(train_loader_emotion):.4f}, "
          f"MSE={total_mse/len(train_loader_emotion):.4f}, "
          f"log_sigma_e={model.log_sigma_emotion.item():.4f}, "
          f"log_sigma_p={model.log_sigma_prosody.item():.4f}")

torch.save(model.state_dict(), "multimodal_uncertainty.pth")

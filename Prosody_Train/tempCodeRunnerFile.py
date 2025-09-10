
# =========================
# 5. Usage Example
# =========================

if __name__ == "__main__":
    audio_dir = "D:\DataSet\Sound_2Dataset\recordings\recordings"  # <-- Update your folder path here
    train_prosody_model(audio_dir, epochs=20, batch_size=4, lr=1e-3)

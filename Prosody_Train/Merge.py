import pandas as pd

# Load your features file
features_df = pd.read_csv(r"D:\DataSet\Sound_2Dataset\features_mp3.csv")

# Load predicted emotions
emotions_df = pd.read_csv("predicted_emotions.csv")

# Merge them on file_name
merged_df = features_df.merge(emotions_df, on="file_name", how="inner")

# Rename the label column for clarity
merged_df = merged_df.rename(columns={"predicted_emotion": "emotion"})

# Save new dataset
merged_df.to_csv("features_with_emotions.csv", index=False)

print("✅ Merged dataset saved as features_with_emotions.csv")
print(merged_df.head())

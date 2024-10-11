import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load and Prepare Data
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)  # Use original sampling rate
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    return features

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def load_data(data_dir):
    X = []
    y = []

    # Loop through each folder in the main directory (Actor_01, Actor_02, etc.)
    for actor_folder in os.listdir(data_dir):
        actor_path = os.path.join(data_dir, actor_folder)

        if os.path.isdir(actor_path):  # Check if it's a directory (actor folder)
            for file_name in os.listdir(actor_path):
                if file_name.endswith('.wav'):  # Only process .wav files
                    audio_file = os.path.join(actor_path, file_name)

                    # Extract features for the current audio file
                    features = extract_features(audio_file)
                    X.append(features)

                    # Extract emotion label from the filename (3rd part)
                    try:
                        emotion_code = file_name.split('-')[2]
                        emotion_label = emotion_map[emotion_code]
                        y.append(emotion_label)
                    except (IndexError, KeyError) as e:
                        print(f"Error processing file {file_name}: {e}")
                        continue  # Skip files with unexpected naming

    return np.array(X), np.array(y)

# Load dataset and preprocess
data_directory = 'Audio_Speech_Actors_01-24'  # Update with your dataset path
X, y = load_data(data_directory)

# Step 2: Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Step 4: Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the Random Forest model
joblib.dump(model, 'emotion_detection_randomForest_model.pkl')

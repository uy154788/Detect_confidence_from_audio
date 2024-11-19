from flask import Flask, request, jsonify
import io
import os
import requests
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment
from collections import defaultdict
import pickle


app = Flask(__name__)

# Load the trained model and label encoder once at the start
with open('emotion_detection_randomForest_model.pkl', 'rb') as f:
    model = pickle.load(f)
# model = joblib.load('emotion_detection_randomForest_model.pkl')
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
label_encoder = LabelEncoder()
label_encoder.fit(emotion_labels)


def video_to_wav(video_file, output_wav):
    video = VideoFileClip(video_file)
    video.audio.write_audiofile(output_wav)


def split_audio(audio_file, chunk_duration=5):
    audio = AudioSegment.from_wav(audio_file)
    chunk_size = chunk_duration * 1000  # chunk duration in milliseconds
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        chunk_buffer = io.BytesIO()
        chunk.export(chunk_buffer, format='wav')
        chunk_buffer.seek(0)
        chunks.append(chunk_buffer)
    return chunks


def extract_features(audio_chunk):
    y, sr = librosa.load(audio_chunk, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    return features


def predict_emotion(chunks):
    emotions = defaultdict(int)  # Dictionary to store emotion frequencies
    for chunk in chunks:
        try:
            features = extract_features(chunk)
            features = features.reshape(1, -1)  # Reshape for prediction
            predicted_emotion = model.predict(features)
            decoded_emotion = label_encoder.inverse_transform([predicted_emotion[0]])[0]
            emotions[decoded_emotion] += 1  # Increment frequency of the predicted emotion
        except Exception as e:
            print(f"Error in processing chunk: {e}")
            emotions['unknown'] += 1  # Store 'unknown' if an error occurs
    return dict(emotions)


def calculate_confidence(emotions):
    confidence_level = 0
    emotion_frame_value = 0
    for emotion, value in emotions.items():
        emotion_frame_value += value
        if emotion == 'happy':
            confidence_level += 0.99 * value
        elif emotion in ['natural', 'calm']:
            confidence_level += 0.95 * value
        elif emotion == 'surprise':
            confidence_level += 0.8 * value
        elif emotion in ['sad', 'fearful']:
            confidence_level += 0.7 * value
        elif emotion in ['angry', 'disgust']:
            confidence_level += 0.6 * value

    confidence = round((confidence_level / emotion_frame_value) * 100, 2)
    return confidence


@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    if 'video_url' not in data:
        return jsonify({'error': 'No video URL provided'}), 400

    video_url = data['video_url']

    try:
        # Download the video from the provided URL
        video_response = requests.get(video_url, stream=True)
        if video_response.status_code != 200:
            return jsonify({'error': 'Failed to download video'}), 400

        # Save the video to a temporary file
        video_file = 'downloaded_video.mp4'
        with open(video_file, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Convert the downloaded video to wav
        output_wav = 'audio.wav'
        video_to_wav(video_file, output_wav)

        # Split the audio into chunks
        chunks = split_audio(output_wav)

        # Predict emotions
        emotions = predict_emotion(chunks)

        # Calculate confidence
        confidence = calculate_confidence(emotions)

        # Clean up the temporary video file
        os.remove(video_file)
        os.remove(output_wav)

        return jsonify({'confidence_level': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port="8003")

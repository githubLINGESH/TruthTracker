import argparse
import numpy as np
import librosa
from keras.models import load_model
import os

def load_model_fn(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, filename)
    return load_model(model_path)

# Function to extract audio features
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Predict function
def main(audio_file_path):
    # Load the model
    loaded_model = load_model_fn('audio_detection_model.h5')

    # Feature extraction
    features = extract_features(audio_file_path)

    # Make a prediction
    prediction = loaded_model.predict(np.array([features]))
    
    if prediction >= 0.5:
        classification = 'REAL'
    else:
        classification = 'FAKE'
    
    return prediction[0], classification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Classification Script")
    parser.add_argument("audio_file_path", type=str, help="Path to the audio file (e.g., wav or mp3)")
    args = parser.parse_args()
    prediction = main(args.audio_file_path)
    print(f"The audio is classified as: {prediction}")

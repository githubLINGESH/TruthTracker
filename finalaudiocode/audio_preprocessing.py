import os
import joblib
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


REAL_FOLDER = 'REAL'
FAKE_FOLDER = 'FAKE'

# Function to extract audio features
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def train_model():
    real_files = [os.path.join(REAL_FOLDER, file) for file in os.listdir(REAL_FOLDER)]
    fake_files = [os.path.join(FAKE_FOLDER, file) for file in os.listdir(FAKE_FOLDER)]

    real_files_encoded = [file.encode('utf-8') for file in real_files]
    fake_files_encoded = [file.encode('utf-8') for file in fake_files]

    real_features = [extract_features(audio_file) for audio_file in real_files_encoded]
    fake_features = [extract_features(audio_file) for audio_file in fake_files_encoded]

    X = np.vstack((real_features, fake_features))
    y = ['real'] * len(real_features) + ['fake'] * len(fake_features)
    
    # Convert labels to floats
    y = np.array([1.0 if label == 'real' else 0.0 for label in y])

    # Define the Keras model
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X, y, epochs=10, batch_size=32)

    return model

# Save the model
def save_model(model, filename):
    model.save(filename)

if __name__ == "__main__":
    # Train the model
    trained_model = train_model()
    
    # Save the model
    save_model(trained_model, 'audio_detection_model.h5')

from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('res.html')



dataset_path = r'C:\Users\india\OneDrive\Desktop\final dataset\Respiratory_Sound_Database\Respiratory_Sound_Database'
max_length = 1723  # Maximum length of the features

def extract_features(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)
    
    # Extract features using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    
    # Pad or truncate the features to a fixed length
    if mel_spectrogram.shape[1] < max_length:
        pad_width = max_length - mel_spectrogram.shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)))
    elif mel_spectrogram.shape[1] > max_length:
        mel_spectrogram = mel_spectrogram[:, :max_length]
    
    return mel_spectrogram


@app.route('/classify-audio', methods=['POST'])
def classify_audio():
    file = request.files['audio']
    file_path = os.path.join(dataset_path, 'audio_and_txt_files', file.filename)
    file.save(file_path)

    # Load the trained model and label encoder
    model = tf.keras.models.load_model(r"C:\Users\india\OneDrive\Desktop\final dataset\Respiratory_Sound_Database\Respiratory_Sound_Database\my_model.h5")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(r"C:\Users\india\OneDrive\Desktop\final dataset\Respiratory_Sound_Database\Respiratory_Sound_Database\label_encoder_classes.npy")

    # Extract features from the audio file
    features = extract_features(file_path)

    # Reshape the input data
    features = features.reshape(1, features.shape[0], features.shape[1], 1)

    # Preprocess the features
    mean = np.mean(features)
    std = np.std(features)
    features = (features - mean) / std

    # Make predictions
    predictions = model.predict(features)

    # Interpret the predictions
    predicted_class_index = np.argmax(predictions)
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

    # Determine respiratory rate
    respiratory_rate_mapping = {
        "AKGC417L": 18,  # Example mapping, replace with your own mapping
        "AKGC417M": 22,
        # Add more mappings for other classes
    }
    respiratory_rate = respiratory_rate_mapping.get(predicted_label)

    # Classify respiratory rate as normal or abnormal
    threshold = 20  # Example threshold, adjust as per your requirements
    if respiratory_rate is not None and respiratory_rate <= threshold:
        classification = "Normal"
    else:
        classification = "Abnormal"

    # Delete the uploaded audio file
    os.remove(file_path)

    return jsonify({'classification': classification})

if __name__ == '__main__':
    app.run(debug=True)

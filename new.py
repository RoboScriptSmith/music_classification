import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("Trained_model.keras")
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        if len(chunk) == 0:
            continue

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

def model_prediction(x_test):
    y_pred = model.predict(x_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, count = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(count)
    max_elements = unique_elements[count == max_count]
    return classes[max_elements[0]]

# Serve the index.html page at the root URL
@app.route('/')
def index():
    return send_from_directory('', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    try:
        # Process the uploaded file
        x_test = load_and_preprocess_data(file_path)
        prediction = model_prediction(x_test)
        
        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

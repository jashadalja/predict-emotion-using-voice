import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd 
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

# Define global variables
fs = 44100  # Default Sampling frequency
model = None
label_encoder = None
loaded_audio = None  # Variable to store loaded or recorded audio data

# Define emotions
emotions = ['Happy', 'Sad', 'Angry', 'Neutral']

# Function to record audio
def record_audio():
    duration = 5  # Duration of recording in seconds

    try:
        print("Recording...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording stopped.")
        return audio.flatten()  # Flatten the array for feature extraction
    except Exception as e:
        show_error("An error occurred while recording: ", e)
        return None

# Function to upload audio file
def upload_audio():
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=(("Audio files", "*.wav"), ("All files", "*.*")),
        initialdir="."
    )
    if file_path:
        try:
            audio_data, sr = librosa.load(file_path, sr=fs)  # Load audio with fixed sampling rate
            if sr != fs:
                show_warning("The uploaded audio file has a different sampling rate. It has been resampled to the default rate.")
            return audio_data
        except Exception as e:
            show_error("An error occurred while uploading the audio file: ", e)
            return None
    else:
        return None

# Function to preprocess audio (extract features)
def preprocess_audio(audio_data):
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=fs, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)  # Normalize MFCCs
        return mfccs_scaled  # Return as a 1D feature vector
    except Exception as e:
        show_error("An error occurred during audio preprocessing: ", e)
        return None

# Function to train the model
def train_model(X, y):
    global model
    global label_encoder
    try:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        model = SVC(kernel='linear', probability=True)
        model.fit(X, y_encoded)

        # Save the model and label encoder
        joblib.dump(model, 'emotion_model.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')

        # Print classification report
        y_pred = model.predict(X)
        print("Training classification report:\n", classification_report(y_encoded, y_pred, target_names=emotions))
        show_info("Model trained successfully.")
    except Exception as e:
        show_error("An error occurred during model training: ", e)

# Function to load the trained model and label encoder
def load_model():
    global model
    global label_encoder
    try:
        model = joblib.load('emotion_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        show_info("Model and label encoder loaded successfully.")
    except Exception as e:
        show_error("An error occurred while loading the model: ", e)

# Function to predict emotion
def predict_emotion(audio_data):
    global model
    global label_encoder
    if model is None or label_encoder is None:
        return "Please train the model first."

    try:
        feature_vector = preprocess_audio(audio_data)
        if feature_vector is not None:
            feature_vector = feature_vector.reshape(1, -1)  # Reshape for model prediction
            emotion_index = model.predict(feature_vector)[0]
            emotion = label_encoder.inverse_transform([emotion_index])[0]
            return emotion
        else:
            return "Error in feature extraction."
    except Exception as e:
        return f"An error occurred during emotion prediction: {e}"

# Function to check if the audio is from a female voice based on pitch
def is_female_voice(audio_data):
    try:
        # Extract pitch (Fundamental Frequency)
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=fs)
        pitch = np.mean([p for p in pitches[pitches > 0]])  # Average pitch

        # Check if the pitch indicates a female voice (average pitch > 165 Hz is a rough heuristic for female voice)
        return pitch > 165
    except Exception as e:
        show_error("An error occurred while detecting the voice gender: ", e)
        return False

# Function to handle record audio button click event
def on_record_button_click():
    global loaded_audio
    loaded_audio = record_audio()
    if loaded_audio is not None:
        if is_female_voice(loaded_audio):
            result_label.config(text="Audio recorded successfully. Now you can predict the emotion.")
        else:
            result_label.config(text="Please record a female voice.")
            loaded_audio = None

# Function to handle upload audio button click event
def on_upload_button_click():
    global loaded_audio
    audio_data = upload_audio()
    if audio_data is not None:
        if is_female_voice(audio_data):
            loaded_audio = audio_data
            result_label.config(text="Audio uploaded successfully. Now you can predict the emotion.")
        else:
            result_label.config(text="Please upload a female voice.")
            loaded_audio = None
    else:
        result_label.config(text="Failed to upload audio. Please try again.")

# Function to handle train model button click event
def on_train_button_click():
    X_train, y_train = load_training_data()  # Load training data (replace with actual data loading)
    train_model(X_train, y_train)

# Function to handle predict button click event
def on_predict_button_click():
    global loaded_audio
    if loaded_audio is None:
        result_label.config(text="Please record or upload audio first.")
    else:
        emotion = predict_emotion(loaded_audio)
        result_label.config(text=f"Predicted emotion: {emotion}")

# Placeholder function to load training data (replace with actual data loading)
def load_training_data():
    # This is a placeholder function. Replace it with code to load your training data.
    # For example, load audio files, extract features, and prepare the data for training.
    # Return feature vectors (X) and labels (y).
    X = np.random.rand(100, 13)  # Example feature matrix
    y = np.random.choice(emotions, size=100)  # Example labels
    return X, y

# Dialog box functions
def show_info(message):
    messagebox.showinfo("Information", message)

def show_warning(message):
    messagebox.showwarning("Warning", message)

def show_error(message, error_details):
    messagebox.showerror("Error", f"{message}\n{error_details}")

# Create main window
root = tk.Tk()
root.title("Emotion Prediction")

# Set window size and background color
root.geometry("600x400")
root.configure(bg="#f0f0f0")

# Create a stylish and bold record button
record_button = tk.Button(
    root,
    text="Record Audio",
    command=on_record_button_click,
    font=("Arial", 16, 'bold'),
    bg='#4CAF50',  # Green background color
    fg='white',    # White text color
    padx=20,       # Padding X
    pady=10,       # Padding Y
    relief='raised', # Raised border for 3D effect
    bd=5,          # Border width
    cursor='hand2'  # Change cursor to hand2 for button click
)
record_button.pack(pady=10)

# Create a stylish and bold upload button
upload_button = tk.Button(
    root,
    text="Upload Audio",
    command=on_upload_button_click,
    font=("Arial", 16, 'bold'),
    bg='#4CAF50',  # Green background color
    fg='white',    # White text color
    padx=20,       # Padding X
    pady=10,       # Padding Y
    relief='raised', # Raised border for 3D effect
    bd=5,          # Border width
    cursor='hand2'  # Change cursor to hand2 for button click
)
upload_button.pack(pady=10)

# Create a stylish and bold train button
train_button = tk.Button(
    root,
    text="Train Model",
    command=on_train_button_click,
    font=("Arial", 16, 'bold'),
    bg='#2196F3',  # Blue background color
    fg='white',    # White text color
    padx=20,       # Padding X
    pady=10,       # Padding Y
    relief='raised', # Raised border for 3D effect
    bd=5,          # Border width
    cursor='hand2'  # Change cursor to hand2 for button click
)
train_button.pack(pady=10)

# Create a stylish and bold predict button
predict_button = tk.Button(
    root,
    text="Predict Emotion",
    command=on_predict_button_click,
    font=("Arial", 16, 'bold'),
    bg='#2196F3',  # Blue background color
    fg='white',    # White text color
    padx=20,       # Padding X
    pady=10,       # Padding Y
    relief='raised', # Raised border for 3D effect
    bd=5,          # Border width
    cursor='hand2'  # Change cursor to hand2 for button click
)
predict_button.pack(pady=10)

# Create result label
result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f0f0f0")
result_label.pack(pady=10)

# Load the trained model and label encoder if they exist
if os.path.exists('emotion_model.pkl') and os.path.exists('label_encoder.pkl'):
    load_model()

# Start GUI main loop
root.mainloop()

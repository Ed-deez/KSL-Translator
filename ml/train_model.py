import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input

# Dynamically set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "..", "data", "processed", "ksl_landmarks.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_FILE = os.path.join(MODEL_DIR, "ksl_cnn_lstm_model.keras")
CLASSES_FILE = os.path.join(MODEL_DIR, "gesture_classes.npy")

def train():
    print("Loading dataset...")
    # Load the dataset (assuming no header row in the CSV)
    df = pd.read_csv(DATA_FILE, header=None)
    
    # Column 0 is the label (e.g., 'Hello', 'A'), Columns 1-63 are the coordinates
    X_raw = df.iloc[:, 1:].values
    y_raw = df.iloc[:, 0].values
    
    # 1. Encode the text labels into numbers (e.g., 'A' -> 0, 'B' -> 1)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    
    # Save the class names so our UI translation page knows what '0' or '1' means
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(CLASSES_FILE, encoder.classes_)
    print(f"Found {num_classes} gesture categories: {encoder.classes_}")

    # 2. Reshape X for the CNN-LSTM Hybrid Model
    # A standard Dense network expects (samples, 63)
    # A Conv1D/LSTM network expects (samples, time_steps, features)
    # Since we have static images right now, time_steps = 1
    X_reshaped = np.reshape(X_raw, (X_raw.shape[0], 1, X_raw.shape[1]))
    
    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.2, random_state=42)
    
    print("Building CNN-LSTM Hybrid Model...")
    # 3. Build the Neural Network Architecture
    model = Sequential([
        Input(shape=(1, 63)),
        
        # CNN Layer: Extracts spatial features between the 21 landmarks
        Conv1D(filters=64, kernel_size=1, activation='relu'),
        
        # LSTM Layer: Captures sequential dependencies
        LSTM(64, return_sequences=False),
        
        # Fully Connected Dense Layers for classification
        Dense(64, activation='relu'),
        Dropout(0.2), # Prevents overfitting by randomly turning off 20% of neurons during training
        Dense(32, activation='relu'),
        
        # Output Layer: Uses softmax to output a confidence percentage for each gesture
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.summary()
    
    # 4. Train the Model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=50, # Number of times it loops through the entire dataset
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    # 5. Evaluate and Save
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
    
    model.save(MODEL_FILE)
    print(f"Model successfully saved to {MODEL_FILE}")

if __name__ == "__main__":
    train()
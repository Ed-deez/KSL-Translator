import os
import cv2
import mediapipe as mp
import csv

# Dynamically set paths relative to the ml/ folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ksl_landmarks.csv")

# Initialize MediaPipe Hands for static images
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, # Crucial for processing individual images instead of a video stream
    max_num_hands=2,
    min_detection_confidence=0.5
)

def extract_and_save():
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: The data directory was not found at {DATA_DIR}")
        print("Please ensure your Kaggle images are placed in 'data/raw/'.")
        return

    print("Starting landmark extraction...")
    
    # Open CSV in write mode
    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Loop through each gesture folder (A, B, C, etc.)
        for label in os.listdir(DATA_DIR):
            label_path = os.path.join(DATA_DIR, label)
            if not os.path.isdir(label_path):
                continue
                
            print(f"Processing gesture: {label}...")
            
            # Loop through every image in the gesture folder
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                
                image = cv2.imread(img_path)
                if image is None:
                    continue # Skip corrupted or unreadable images
                
                # Convert BGR to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                # If hands are found, save their coordinates
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        row = [label] # The first column is the target label (the answer)
                        
                        for landmark in hand_landmarks.landmark:
                            row.extend([landmark.x, landmark.y, landmark.z])
                            
                        writer.writerow(row)
                        
    print(f"Extraction complete! Data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_and_save()
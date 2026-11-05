import os
import cv2
import mediapipe as mp
import csv
import pandas as pd

# Dynamically set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw", "Images")
TRAIN_CSV = os.path.join(BASE_DIR, "..", "data", "raw", "train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ksl_landmarks.csv")

def extract_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: Could not find {TRAIN_CSV}. Please place it in data/raw/")
        return

    print("Loading answer key from train.csv...")
    # Load the CSV containing the img_IDS and Label columns
    df = pd.read_csv(TRAIN_CSV)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    print("Starting landmark extraction. This may take a few minutes...")
    
    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Loop through every row in train.csv
        for index, row in df.iterrows():
            img_id = str(row['img_IDS'])
            label = row['Label']
            
            # If the CSV doesn't include the file extension, append .jpg
            if not img_id.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_id += '.jpg'
                
            img_path = os.path.join(DATA_DIR, img_id)
            
            # Skip if the image wasn't found (e.g., if it was deleted during our cleaning step)
            if not os.path.exists(img_path):
                continue
                
            image = cv2.imread(img_path)
            if image is None:
                continue 
            
            # Convert and process the image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            # Save the true label alongside the coordinates
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row_data = [label] 
                    for landmark in hand_landmarks.landmark:
                        row_data.extend([landmark.x, landmark.y, landmark.z])
                        
                    writer.writerow(row_data)
                    
    print(f"Extraction complete! Properly labeled data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_and_save()
import cv2
import mediapipe as mp
import os
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Set up your paths
DATA_DIR = '../data/raw'
OUTPUT_FILE = '../data/processed/ksl_landmarks.csv'

def process_images():
    # Open a CSV to save our data
    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Loop through each folder (assuming folders are named by KSL sign, e.g., 'A', 'B')
        for label_dir in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, label_dir)
            if not os.path.isdir(path):
                continue
                
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue

                # MediaPipe needs RGB, OpenCV uses BGR
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract the 21 x,y,z coordinates into a flat list
                        row = [label_dir] # First column is the KSL sign/label
                        for landmark in hand_landmarks.landmark:
                            row.extend([landmark.x, landmark.y, landmark.z])
                        
                        # Save to CSV
                        writer.writerow(row)

if __name__ == "__main__":
    print("Starting extraction...")
    process_images()
    print("Extraction complete. Landmarks saved to CSV.")
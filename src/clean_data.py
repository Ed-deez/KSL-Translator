import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw", "Images")

def clean_dataset():
    removed_count = 0
    
    for label_dir in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, label_dir)
        if not os.path.isdir(path):
            continue
            
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            
            # 1. Delete hidden system files and non-images
            if img_name.startswith('.') or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Removing invalid file type: {img_name}")
                os.remove(img_path)
                removed_count += 1
                continue
                
            # 2. Check for image corruption
            # If OpenCV cannot read it, it returns None
            image = cv2.imread(img_path)
            if image is None:
                print(f"Removing corrupted image: {img_name}")
                os.remove(img_path)
                removed_count += 1
                
    print(f"Cleaning complete. Removed {removed_count} unusable files.")

if __name__ == "__main__":
    clean_dataset()
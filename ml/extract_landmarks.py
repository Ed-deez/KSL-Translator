
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "kslc-dataset"
TRAIN_CSV = DATASET_DIR / "Train.csv"
IMAGE_DIR = DATASET_DIR / "Images" / "Images"

MODEL_PATH = PROJECT_ROOT / "assets" / "hand_landmarker.task"
OUTPUT_CSV = PROJECT_ROOT / "data" / "landmarks" / "kslc_landmarks.csv"


def find_image(image_id):
    """Find an image using its ID, with or without .jpg."""
    image_id = str(image_id).strip()

    candidates = [
        IMAGE_DIR / image_id,
        IMAGE_DIR / f"{image_id}.jpg",
        IMAGE_DIR / f"{image_id}.jpeg",
        IMAGE_DIR / f"{image_id}.png",
    ]

    for path in candidates:
        if path.is_file():
            return path

    return None


def extract_normalized_landmarks(result):
    """Convert 21 hand landmarks into 63 normalized features."""
    if not result.hand_landmarks:
        return None

    points = np.array(
        [
            [point.x, point.y, point.z]
            for point in result.hand_landmarks[0]
        ],
        dtype=np.float32,
    )

    # Make the wrist the origin.
    points -= points[0].copy()

    # Scale coordinates to reduce differences in hand size.
    scale = np.max(np.abs(points))

    if scale > 0:
        points /= scale

    return points.flatten()


def main():
    if not TRAIN_CSV.is_file():
        raise FileNotFoundError(
            f"Training CSV not found: {TRAIN_CSV}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH}"
        )

    train_df = pd.read_csv(TRAIN_CSV)

    required_columns = {"img_IDS", "Label"}
    if not required_columns.issubset(train_df.columns):
        raise ValueError(
            "Train.csv must contain columns: img_IDS and Label. "
            f"Found: {list(train_df.columns)}"
        )

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    feature_columns = [
        f"f{i}" for i in range(63)
    ]

    records = []
    missing_images = 0
    unreadable_images = 0
    no_hand_detected = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

        total = len(train_df)

        for index, row in train_df.iterrows():
            image_id = row["img_IDS"]
            label = row["Label"]

            image_path = find_image(image_id)

            if image_path is None:
                missing_images += 1
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                unreadable_images += 1
                continue

            rgb_image = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image,
            )

            result = landmarker.detect(mp_image)
            features = extract_normalized_landmarks(result)

            if features is None:
                no_hand_detected += 1
                continue

            record = dict(zip(feature_columns, features))
            record["label"] = str(label)
            record["image_id"] = str(image_id)

            records.append(record)

            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1}/{total} images")

    OUTPUT_CSV.parent.mkdir(
        parents=True, exist_ok=True
    )

    output_columns = (
        feature_columns + ["label", "image_id"]
    )

    output_df = pd.DataFrame(
        records, columns=output_columns
    )
    output_df.to_csv(OUTPUT_CSV, index=False)

    print("\nExtraction complete.")
    print(f"Images in training CSV: {len(train_df)}")
    print(f"Landmarks extracted: {len(records)}")
    print(f"Missing images: {missing_images}")
    print(f"Unreadable images: {unreadable_images}")
    print(f"No hand detected: {no_hand_detected}")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
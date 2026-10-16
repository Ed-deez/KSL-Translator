
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


CSV_PATH = "data/landmarks/ksl_landmarks.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR, "ksl_model.keras"
)

LABELS_PATH = os.path.join(
    MODEL_DIR, "labels.json"
)

FEATURE_COLUMNS = [
    f"f{i}" for i in range(63)
]


def main():

    # 1. Load the collected data.
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            "Dataset not found. Run "
            "'python -m ml.collect_data' first."
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = (
        FEATURE_COLUMNS + ["label", "session_id"]
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values. "
            "Check the CSV before training."
        )

    if df["label"].nunique() < 2:
        raise ValueError(
            "Collect at least two different sign classes."
        )

    if df["session_id"].nunique() < 3:
        raise ValueError(
            "Collect at least three separate sessions, "
            "with every sign represented in each session. "
            "This lets us test on a session the model "
            "has not trained on."
        )

    print("\nDataset loaded successfully.")
    print("Total examples:", len(df))
    print("\nExamples per class:")
    print(df["label"].value_counts())
    print("\nCollection sessions:", df["session_id"].nunique())

    # 2. Separate features and labels.
    X = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(
        df["label"].astype(str)
    )

    class_names = label_encoder.classes_.tolist()

    # 3. Split by collection session.
    # Entire sessions are held out to reduce leakage
    # from nearly identical neighbouring samples.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=df["session_id"]
        )
    )

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train_int = y[train_indices]
    y_test_int = y[test_indices]

    # Every class must occur in both partitions.
    missing_train = set(
        np.unique(y)
    ) - set(
        np.unique(y_train_int)
    )

    missing_test = set(
        np.unique(y)
    ) - set(
        np.unique(y_test_int)
    )

    if missing_train or missing_test:
        raise ValueError(
            "The session split omitted one or more classes. "
            "Make sure every sign is collected in every "
            "session, then run training again."
        )

    print("\nTraining examples:", len(X_train))
    print("Test examples:", len(X_test))
    print("Classes:", class_names)

    # 4. Convert class numbers into one-hot labels.
    y_train = to_categorical(
        y_train_int,
        num_classes=len(class_names)
    )

    # 5. Build the neural network.
    model = Sequential([
        Input(shape=(63,)),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.2),

        Dense(
            len(class_names),
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 6. Stop training if validation performance
    # stops improving.
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # 7. Train using only the training sessions.
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # 8. Evaluate on the held-out session(s).
    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test_int,
        predictions
    )

    print("\nHeld-out session accuracy:")
    print(f"{accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification report:")
    print(
        classification_report(
            y_test_int,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test_int,
            predictions,
            labels=list(range(len(class_names)))
        )
    )

    # 9. Save the trained model and its label mapping.
    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(MODEL_PATH)

    with open(
        LABELS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            class_names,
            file,
            indent=4
        )

    print("\nTraining complete.")
    print("Saved model:", MODEL_PATH)
    print("Saved labels:", LABELS_PATH)


if __name__ == "__main__":
    main()

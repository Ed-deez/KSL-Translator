import os
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "..", "data", "processed", "ksl_landmarks.csv")

def impute_missing_values():
    print(f"Loading dataset from {DATA_FILE}...")
    
    # Load the extracted dataset
    df = pd.read_csv(DATA_FILE, header=None)
    
    # Check for missing values before imputation
    missing_before = df.isnull().sum().sum()
    print(f"Total missing values before imputation: {missing_before}")
    
    if missing_before > 0:
        print("Imputing missing coordinate data using column means...")
        # Separate the labels (column 0) from the numerical coordinate data (columns 1-63)
        labels = df.iloc[:, 0]
        features = df.iloc[:, 1:]
        
        # Fill missing values (NaN) with the mean of their respective coordinate column
        features = features.fillna(features.mean())
        
        # Recombine the data and save it back to the CSV
        df_cleaned = pd.concat([labels, features], axis=1)
        df_cleaned.to_csv(DATA_FILE, index=False, header=False)
        
        print("Imputation complete. The dataset is now dense and ready for training.")
    else:
        print("Dataset is already complete. No imputation necessary.")

if __name__ == "__main__":
    impute_missing_values()
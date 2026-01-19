import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from src.ingestion.stream_loader import StreamLoader
from src.processing.preprocessor import Preprocessor
from src.models.baseline import IsolationForestModel
from src.models.deep_learning import AutoEncoderModel, LSTMModel

def evaluate_model(model_name, y_true, y_pred, y_prob=None):
    print(f"\n--- {model_name} Evaluation ---")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # If probabilistic predictions available
    # if y_prob is not None:
    #     print(f"ROC-AUC: {roc_auc_score(y_true, y_prob)}")

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def main():
    # 1. Setup & Load Data
    print("Loading data...")
    # Use the smaller testing set for quick training/verification in this simulation
    # In production, training set would be UNSW-NB15_1.csv etc.
    train_file = r"D:\DS\Anomaly Detection\DATA\UNSW_NB15_testing-set.csv" 
    # Logic: We usually train on "Normal" data for anomaly detection or a mix with labels.
    # The dataset has labels. 
    
    loader = StreamLoader(train_file, chunk_size=100000) 
    # Load all data for simulation simplicity (or first chunk)
    df = next(loader.stream())
    
    print(f"Loaded {df.shape[0]} samples.")

    # 2. Preprocessing
    print("Preprocessing...")
    preprocessor = Preprocessor()
    preprocessor.fit(df)
    X, y = preprocessor.transform(df)
    
    # Split into Train/Test (Simulating historical vs new data)
    # Using 80/20 split
    split_idx = int(X.shape[0] * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # For Unsupervised Anomaly Detection (Isolation Forest / Autoencoder), 
    # we often train on 'Normal' data only, or assume outliers are rare in training set.
    # Let's see if we can filter Normal for training Autoencoder best practice.
    # label 0 = Normal (assuming based on typical datasets, check EDA output to be sure)
    # If y is available, we can refine training data.
    
    # Filter Training Data for Deep Learning (Normal only often better for AE)
    # Assuming 0 is Normal.
    train_mask = (y_train == 0)
    X_train_normal = X_train[train_mask]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Normal training samples: {X_train_normal.shape[0]}")

    # 3. Model 1: Isolation Forest (Baseline)
    print("\nTraining Isolation Forest...")
    iso_forest = IsolationForestModel(contamination=0.05) # Estimate 5% anomalies
    iso_forest.fit(X_train) # IF can handle outliers in training
    
    # Predict
    iso_preds = iso_forest.predict(X_test)
    evaluate_model("Isolation Forest", y_test, iso_preds)

    # 4. Model 2: Autoencoder
    print("\nTraining Autoencoder...")
    input_dim = X_train.shape[1]
    autoencoder = AutoEncoderModel(input_dim=input_dim)
    # Train on normal data primarily for reconstruction error logic
    # If X_train_normal is too small, use X_train but AE might learn anomalies
    ae_train_data = X_train_normal if X_train_normal.shape[0] > 1000 else X_train
    autoencoder.fit(ae_train_data, epochs=5, batch_size=64)
    
    ae_preds = autoencoder.predict(X_test)
    evaluate_model("Autoencoder", y_test, ae_preds)

    # 5. Model 3: LSTM (Sequence based)
    # create sequences
    print("\nTraining LSTM...")
    TIME_STEPS = 5
    # We need to reshape X_train/test into sequences
    # Note: this reduces sample size by TIME_STEPS
    # For simplicity, we regenerate sequences from the raw split X
    
    X_seq_train, y_seq_train = create_sequences(X_train, y_train, TIME_STEPS)
    X_seq_test, y_seq_test = create_sequences(X_test, y_test, TIME_STEPS)
    
    # Train on normal sequences only
    train_seq_mask = (y_seq_train == 0)
    X_seq_train_normal = X_seq_train[train_seq_mask]
    
    lstm_model = LSTMModel(timesteps=TIME_STEPS, n_features=input_dim)
    lstm_train_data = X_seq_train_normal if X_seq_train_normal.shape[0] > 1000 else X_seq_train
    lstm_model.fit(lstm_train_data, epochs=5, batch_size=64)
    
    lstm_preds = lstm_model.predict(X_seq_test)
    evaluate_model("LSTM", y_seq_test, lstm_preds)

    # --- SAVE MODELS ---
    print("\nSaving models...")
    # Ensure directories exist
    os.makedirs("src/models", exist_ok=True)
    os.makedirs("src/processing", exist_ok=True)

    preprocessor.save("src/processing/preprocessor.pkl")
    iso_forest.save("src/models/isolation_forest.pkl")
    autoencoder.save("src/models/autoencoder")
    # lstm_model.save("src/models/lstm") # Optional
    print("Models saved successfully.")

    # 6. Prediction / Alerting Simulation (Showcase)
    print("\n--- Alert Simulation (Test Stream) ---")
    print("Streaming last 10 test samples...")
    for i in range(10):
        sample = X_test[i:i+1] # 2D array
        label = y_test[i]
        
        # Ensemble Logic? Or just print individuals
        if_anom = iso_forest.predict(sample)[0]
        ae_anom = autoencoder.predict(sample)[0]
        
        status = "CRITICAL" if (if_anom == 1 and ae_anom == 1) else "WARNING" if (if_anom == 1 or ae_anom == 1) else "Normal"
        actual = "Anomaly" if label == 1 else "Normal"
        
        print(f"Sample {i}: Actual={actual} | IF={if_anom}, AE={ae_anom} | System Alert: {status}")

    print("\nSystem verification complete.")

if __name__ == "__main__":
    main()

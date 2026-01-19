import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import joblib
import os

class AutoEncoderModel:
    """
    Autoencoder for anomaly detection.
    Trains on normal data to minimize reconstruction error.
    Anomalies are detected when reconstruction error is high.
    """
    def __init__(self, input_dim, encoding_dim=14):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = self._build_model()
        self.threshold = None

    def _build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        output_layer = Dense(self.input_dim, activation='sigmoid')(decoded) # Assuming MinMax scaled data (0-1)
        # If StandardScaled, maybe linear activation is better for output, or stick to relu/linear.
        # Let's use linear for flexibility with StandardScaler
        output_layer = Dense(self.input_dim, activation='linear')(decoded)

        autoencoder = Model(input_layer, output_layer)
        autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        return autoencoder

    def fit(self, X, epochs=10, batch_size=32, validation_split=0.1):
        print(f"Training Autoencoder with input shape {X.shape}...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Determine threshold based on training data reconstruction error
        reconstructions = self.model.predict(X)
        loss = np.mean(np.abs(reconstructions - X), axis=1)
        self.threshold = np.mean(loss) + 2 * np.std(loss) # Simple threshold: mean + 2*std
        print(f"Autoencoder training complete. Threshold set to: {self.threshold}")
        return history

    def predict(self, X):
        """
        Predict anomalies.
        Returns: 1 for anomaly, 0 for normal (based on threshold).
        """
        reconstructions = self.model.predict(X)
        loss = np.mean(np.abs(reconstructions - X), axis=1)
        return (loss > self.threshold).astype(int)

    def save(self, path="autoencoder"):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, "model.h5"))
        joblib.dump(self.threshold, os.path.join(path, "threshold.pkl"))
        print(f"Autoencoder saved to {path}")

    def load(self, path="autoencoder"):
        self.model = tf.keras.models.load_model(os.path.join(path, "model.h5"))
        self.threshold = joblib.load(os.path.join(path, "threshold.pkl"))
        print(f"Autoencoder loaded from {path}")


class LSTMModel:
    """
    LSTM for sequence-based anomaly detection.
    Predicts the next step (or reconstructs sequence). 
    Here we implement a reconstruction LSTM Autoencoder or a predictor.
    Let's go with LSTM Autoencoder for sequence reconstruction.
    """
    def __init__(self, timesteps, n_features):
        self.timesteps = timesteps
        self.n_features = n_features
        self.model = self._build_model()
        self.threshold = None

    def _build_model(self):
        model = Sequential()
        # Encoder
        model.add(LSTM(64, activation='relu', input_shape=(self.timesteps, self.n_features), return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(RepeatVector(self.timesteps))
        # Decoder
        model.add(LSTM(32, activation='relu', return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(self.n_features)))
        
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        return model

    def fit(self, X, epochs=10, batch_size=32, validation_split=0.1):
        """
        X shape must be (n_samples, timesteps, n_features)
        """
        print(f"Training LSTM with input shape {X.shape}...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        reconstructions = self.model.predict(X)
        loss = np.mean(np.abs(reconstructions - X), axis=(1, 2))
        self.threshold = np.mean(loss) + 2 * np.std(loss)
        print(f"LSTM training complete. Threshold set to: {self.threshold}")
        return history

    def predict(self, X):
        reconstructions = self.model.predict(X)
        loss = np.mean(np.abs(reconstructions - X), axis=(1, 2))
        return (loss > self.threshold).astype(int)

    def save(self, path="lstm"):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, "model.h5"))
        joblib.dump(self.threshold, os.path.join(path, "threshold.pkl"))
        print(f"LSTM saved to {path}")

    def load(self, path="lstm"):
        self.model = tf.keras.models.load_model(os.path.join(path, "model.h5"))
        self.threshold = joblib.load(os.path.join(path, "threshold.pkl"))
        print(f"LSTM loaded from {path}")

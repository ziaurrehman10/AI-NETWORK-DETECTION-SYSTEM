from sklearn.ensemble import IsolationForest
import joblib
import numpy as np

class IsolationForestModel:
    """
    Wrapper for Isolation Forest model for Anomaly Detection.
    """
    def __init__(self, contamination=0.01, random_state=42):
        """
        Args:
            contamination (float): Expected proportion of outliers in the data set.
        """
        self.model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
        self.is_fitted = False

    def fit(self, X):
        """
        Fit the model on normal data (or mixed data assuming anomalies are rare).
        """
        print(f"Training Isolation Forest with input shape {X.shape}...")
        self.model.fit(X)
        self.is_fitted = True
        print("Isolation Forest training complete.")

    def predict(self, X):
        """
        Predict anomalies.
        Returns:
            np.array: 1 for normal, -1 for anomaly.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        # sklearn IsolationForest returns -1 for outliers and 1 for inliers.
        # We might want to map this to: 0 for normal, 1 for anomaly to match dataset label convention?
        # Dataset: 0 = Normal, 1 = Attack (usually). 
        # Check EDA output to confirm label semantics. Assuming 0 is normal.
        preds = self.model.predict(X)
        
        # Map: 1 (inlier) -> 0 (Normal), -1 (outlier) -> 1 (Anomaly)
        mapped_preds = np.where(preds == 1, 0, 1)
        return mapped_preds

    def save(self, path="isolation_forest.pkl"):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path="isolation_forest.pkl"):
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"Model loaded from {path}")

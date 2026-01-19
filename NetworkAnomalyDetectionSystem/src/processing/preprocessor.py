import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class Preprocessor:
    """
    Handles data cleaning, encoding, and normalization.
    """
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        # Define categorical columns based on UNSW-NB15 dataset description
        # Note: 'attack_cat' is the multiclass label, 'label' is binary.
        # We might want to ENCODE 'attack_cat' if it's used as a feature, 
        # but usually it's a target. For anomaly detection (unsupervised), we ignore targets.
        # But if we use it, we must encode it.
        self.categorical_cols = ['proto', 'service', 'state', 'attack_cat']
        self.drop_cols = ['id'] 
        self.is_fitted = False

    def fit(self, df):
        """
        Fits encoders and scaler on the provided dataframe.
        """
        df_clean = df.drop(columns=self.drop_cols, errors='ignore')
        
        # Fit Label Encoders
        for col in self.categorical_cols:
            if col in df_clean.columns:
                le = LabelEncoder()
                # Convert to string to ensure uniformity
                val_series = df_clean[col].astype(str)
                le.fit(val_series)
                self.label_encoders[col] = le
        
        # Fit Scaler on numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        # Exclude labels from scaling
        numerical_cols = [c for c in numerical_cols if c not in ['label', 'attack_cat'] and c not in self.categorical_cols]
        
        if numerical_cols:
            self.scaler.fit(df_clean[numerical_cols])
        self.numerical_cols = numerical_cols
        
        self.is_fitted = True
        print("Preprocessor fitted.")

    def transform(self, df):
        """
        Transforms the dataframe.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted using .fit() before .transform()")

        df_clean = df.copy()
        df_clean = df_clean.drop(columns=self.drop_cols, errors='ignore')

        # Transform Categorical
        for col, le in self.label_encoders.items():
            if col in df_clean.columns:
                # Robust transform with handling for unseen labels
                # Convert to string
                vals = df_clean[col].astype(str)
                
                # Identify unseen
                classes_set = set(le.classes_)
                # internal function to map safe
                def map_safe(x):
                    return x if x in classes_set else list(classes_set)[0] 
                
                # Apply map (can be slow for huge data, but robust)
                vals_mapped = vals.apply(map_safe)
                
                df_clean[col] = le.transform(vals_mapped)

        # Separate Label if exists
        y = None
        if 'label' in df_clean.columns:
            y = df_clean['label'].values
        
        # Transform Numerical
        if self.numerical_cols:
            # fill missing numericals with 0 just in case
            df_clean[self.numerical_cols] = df_clean[self.numerical_cols].fillna(0)
            df_clean[self.numerical_cols] = self.scaler.transform(df_clean[self.numerical_cols])
        
        # X should include encoded categoricals + scaled numericals
        # We must ensure X only contains features the model expects.
        # DROP Targets from X
        X_df = df_clean.drop(columns=['label', 'attack_cat'], errors='ignore')
        
        # If 'attack_cat' was in encoded columns, it might still be there if we didn't drop it.
        # But usually attack_cat is target. We shouldn't use it for detecting anomalies in unsupervised approach theoretically,
        # unless it's a feature in this dataset (it's the fine-grained label).
        # We should DROP it from X.
        
        X = X_df.values
        
        return X, y

    def save(self, path="preprocessor.pkl"):
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path="preprocessor.pkl"):
        return joblib.load(path)

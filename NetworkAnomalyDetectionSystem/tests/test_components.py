import unittest
import pandas as pd
import numpy as np
import os
import sys
import shutil

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.stream_loader import StreamLoader
from src.processing.preprocessor import Preprocessor
from src.models.baseline import IsolationForestModel
from src.models.deep_learning import AutoEncoderModel

class TestComponents(unittest.TestCase):
    
    def setUp(self):
        # Create dummy CSV
        self.dummy_csv = "dummy.csv"
        df = pd.DataFrame({
            'id': range(10),
            'dur': np.random.rand(10),
            'proto': ['tcp'] * 5 + ['udp'] * 5,
            'service': ['http'] * 10,
            'state': ['CON'] * 10,
            'spkts': np.random.randint(1, 10, 10),
            'dpkts': np.random.randint(1, 10, 10),
            'sbytes': np.random.randint(100, 1000, 10),
            'label': [0] * 8 + [1] * 2,
            'attack_cat': ['Normal'] * 8 + ['Generic'] * 2
        })
        df.to_csv(self.dummy_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.dummy_csv):
            os.remove(self.dummy_csv)
            
        if os.path.exists("test_preprocessor.pkl"):
            os.remove("test_preprocessor.pkl")
            
        if os.path.exists("test_iso_forest.pkl"):
            os.remove("test_iso_forest.pkl")
            
        if os.path.exists("test_ae_model"):
            shutil.rmtree("test_ae_model")

    def test_stream_loader(self):
        loader = StreamLoader(self.dummy_csv, chunk_size=5)
        chunks = list(loader.stream())
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 5)

    def test_preprocessor(self):
        df = pd.read_csv(self.dummy_csv)
        preprocessor = Preprocessor()
        preprocessor.fit(df)
        
        X, y = preprocessor.transform(df)
        self.assertEqual(X.shape[0], 10)
        self.assertEqual(y.shape[0], 10)
        
        # Test persistence
        preprocessor.save("test_preprocessor.pkl")
        self.assertTrue(os.path.exists("test_preprocessor.pkl"))
        
        loaded_prep = Preprocessor.load("test_preprocessor.pkl")
        self.assertTrue(loaded_prep.is_fitted)

    def test_isolation_forest(self):
        # Prepare data
        df = pd.read_csv(self.dummy_csv)
        prep = Preprocessor()
        prep.fit(df)
        X, y = prep.transform(df)
        
        model = IsolationForestModel(contamination=0.1)
        model.fit(X)
        
        preds = model.predict(X)
        self.assertEqual(len(preds), 10)
        
        # Test persistence
        model.save("test_iso_forest.pkl")
        self.assertTrue(os.path.exists("test_iso_forest.pkl"))
        
        model.load("test_iso_forest.pkl")
        self.assertTrue(model.is_fitted)

    def test_autoencoder(self):
        try:
            df = pd.read_csv(self.dummy_csv)
            prep = Preprocessor()
            prep.fit(df)
            X, y = prep.transform(df)
            
            input_dim = X.shape[1]
            model = AutoEncoderModel(input_dim=input_dim)
            # Mock training for speed
            model.fit(X, epochs=1, batch_size=2)
            
            preds = model.predict(X)
            self.assertEqual(len(preds), 10)
            
            # Save/Load
            save_dir = "test_ae_model"
            # Debug print
            print(f"Saving to {save_dir}")
            model.save(save_dir)
            self.assertTrue(os.path.exists(save_dir))
            
            new_model = AutoEncoderModel(input_dim=input_dim)
            new_model.load(save_dir)
            self.assertIsNotNone(new_model.threshold)
            
            # Cleanup
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

if __name__ == '__main__':
    unittest.main()

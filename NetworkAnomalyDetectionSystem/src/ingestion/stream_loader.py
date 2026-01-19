import pandas as pd
import time
import os

class StreamLoader:
    """
    Simulates a data stream from a CSV file.
    Reads data in chunks to mimic real-time network traffic.
    """
    def __init__(self, file_path, chunk_size=1000, delay=0.0):
        """
        Args:
            file_path (str): Path to the CSV file.
            chunk_size (int): Number of rows to yield per chunk.
            delay (float): Artificial delay in seconds between chunks to simulate streaming latency.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.delay = delay
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    def stream(self):
        """
        Yields chunks of data from the CSV.
        """
        print(f"Starting stream from {self.file_path}...")
        # Use pandas chunksize to read iteratively
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            if self.delay > 0:
                time.sleep(self.delay)
            yield chunk
        print("Stream finished.")

if __name__ == "__main__":
    # Test the loader
    data_path = r"D:\DS\Anomaly Detection\DATA\UNSW_NB15_testing-set.csv"
    loader = StreamLoader(data_path, chunk_size=5)
    
    print("Testing StreamLoader...")
    for i, batch in enumerate(loader.stream()):
        print(f"Batch {i+1}: {batch.shape}")
        print(batch.head(2))
        if i >= 2: break # Stop after 3 batches for testing

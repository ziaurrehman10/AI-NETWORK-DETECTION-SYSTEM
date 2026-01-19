import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.stream_loader import StreamLoader

def run_eda(file_path):
    print("Starting EDA...")
    loader = StreamLoader(file_path, chunk_size=50000) # Load a large chunk for EDA
    
    # Get first chunk
    df = next(loader.stream())
    
    print("\nData Shape (Sample):", df.shape)
    print("\nColumn Types:\n", df.dtypes)
    
    print("\nMissing Values:\n", df.isnull().sum().sum())
    
    print("\nClass Distribution (attack_cat):\n", df['attack_cat'].value_counts())
    print("\nLabel Distribution (0/1):\n", df['label'].value_counts())
    
    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df['attack_cat'], order=df['attack_cat'].value_counts().index)
    plt.title('Attack Category Distribution (Sample)')
    plt.savefig('notebooks/attack_distribution.png')
    print("\nSaved attack_distribution.png")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('notebooks/correlation_matrix.png')
    print("Saved correlation_matrix.png")

if __name__ == "__main__":
    data_path = r"D:\DS\Anomaly Detection\DATA\UNSW_NB15_testing-set.csv"
    # Fallback or check other files if needed
    if not os.path.exists(data_path):
        data_path = r"D:\DS\Anomaly Detection\DATA\UNSW-NB15_1.csv"
    
    run_eda(data_path)

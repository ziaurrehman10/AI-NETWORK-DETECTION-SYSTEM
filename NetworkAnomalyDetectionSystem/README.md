# Network Anomaly Detection System

A comprehensive system for detecting network anomalies using machine learning (Isolation Forest) and deep learning (Autoencoder, LSTM). The system includes real-time data ingestion simulation, preprocessing, model training, and an interactive dashboard for monitoring.

## 📂 Project Structure

```
.
├── main.py                     # Orchestration script for training and evaluation
├── requirements.txt            # Python dependencies
├── src/
│   ├── dashboard/
│   │   └── app.py              # Streamlit dashboard application
│   ├── ingestion/
│   │   └── stream_loader.py    # Simulates real-time data streaming
│   ├── models/
│   │   ├── baseline.py         # Isolation Forest model
│   │   └── deep_learning.py    # Autoencoder and LSTM models
│   └── processing/
│       └── preprocessor.py     # Data cleaning and scaling
├── notebooks/                  # Exploratory Data Analysis (EDA)
└── reports/                    # Generated reports
```

## 🚀 Setup & Installation

1.  **Clone the repository** (if applicable) or ensure you have the source code.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### 1. Train Models
Before running the dashboard, you must train the models and save the artifacts (preprocessor, models).

```bash
python main.py
```
This script will:
*   Load the dataset.
*   Preprocess the data.
*   Train Isolation Forest, Autoencoder, and LSTM models.
*   Evaluate the models.
*   **Note**: Ensure the line `iso_forest.save(...)`, `preprocessor.save(...)`, etc., are present in `main.py` if they are not already called (based on current code state, they might need to be verified).

### 2. Run the Dashboard
Launch the real-time monitoring dashboard:

```bash
streamlit run src/dashboard/app.py
```

*   Use the sidebar to control simulation speed and batch size.
*   Click **Start Monitoring** to begin the simulation.

## 🧠 Models Implemented

*   **Isolation Forest**: Unsupervised algorithm that isolates anomalies by randomly selecting a feature and then randomly selecting a split value.
*   **Autoencoder**: Neural network that learns to reconstruct normal data; high reconstruction error indicates anomalies.
*   **LSTM**: Long Short-Term Memory network for detecting anomalies in time-series sequences.

## 📊 Dataset
The system is designed to work with the **UNSW-NB15** dataset (or compatible network traffic CSVs). Ensure the data path in `main.py` and `src/dashboard/app.py` points to your local dataset file.

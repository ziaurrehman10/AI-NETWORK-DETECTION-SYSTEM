# Network Anomaly Detection System - Documentation

## 1. System Overview
The **Network Anomaly Detection System** is a real-time monitoring solution designed to identify malicious network traffic patterns using a hybrid approach of machine learning and deep learning. The system simulates network traffic ingestion, processes data in real-time, and applies an ensemble of models to flag anomalies.

### Key Features
*   **Real-time Simulation**: Ingests CSV data as a live stream.
*   **Hybrid Detection**: Combines statistical (Isolation Forest) and neural methods (Autoencoder, LSTM).
*   **Live Dashboard**: Interactive visualization of traffic volume and anomaly scores.
*   **Robust Persistence**: Metrics, data scaling, and trained models are fully serializable.

---

## 2. Architecture

The system follows a modular pipeline architecture:

```mermaid
graph LR
    A[Data Source (CSV)] -->|StreamLoader| B(Ingestion)
    B -->|Chunks| C(Preprocessor)
    C -->|Scaled Data| D{Model Ensemble}
    D -->|Prediction| E[Isolation Forest]
    D -->|Prediction| F[Autoencoder]
    D -->|Prediction| G[LSTM]
    E & F & G -->|Aggregated Score| H[Dashboard / Alerting]
```

---

## 3. Component Reference

### 3.1 Ingestion Layer (`src/ingestion`)
*   **`StreamLoader`**: 
    *   **Purpose**: Simulates a live data feed from static datasets.
    *   **Mechanism**: Reads large CSV files in configurable chunks (batches) and yields them with an optional time delay to mimic network latency.
    *   **Key Methods**: `stream()` (generator).

### 3.2 Processing Layer (`src/processing`)
*   **`Preprocessor`**:
    *   **Purpose**: Prepares raw packet data for model consumption.
    *   **Features**:
        *   **Label Encoding**: Converts categorical fields (`proto`, `service`, `state`) to numeric representations. Handles unseen labels during inference.
        *   **Scaling**: Standardizes numerical features (mean=0, variance=1) using `StandardScaler`.
        *   **Persistence**: Saves fitted state to `preprocessor.pkl`.

### 3.3 Modeling Layer (`src/models`)
The system expects trained models to be saved in `src/models/`.

#### Baseline: Isolation Forest (`baseline.py`)
*   **Algorithm**: `sklearn.ensemble.IsolationForest`.
*   **Logic**: Builds random trees; anomalies are isolated closer to the root (shorter path length).
*   **Output**: Binary (0 = Normal, 1 = Anomaly).

#### Deep Learning: Autoencoder (`deep_learning.py`)
*   **Architecture**: Feed-forward dense neural network.
    *   **Encoder**: Compresses input to low-dimensional latent space.
    *   **Decoder**: Reconstructs input from latent space.
*   **Logic**: Trained on normal data to minimize **Mean Squared Error (MSE)**. High reconstruction error during inference implies an anomaly.
*   **Thresholding**: Dynamic threshold set at $\mu + 2\sigma$ of training loss.

#### Deep Learning: LSTM (`deep_learning.py`)
*   **Architecture**: Long Short-Term Memory network.
*   **logic**: Captures temporal dependencies in packet sequences. Ideal for detecting sequential attacks (e.g., DoS floods).

### 3.4 Visualization Layer (`src/dashboard`)
*   **`app.py`**: A **Streamlit** application.
    *   **Real-time Loop**: Consumes the `StreamLoader` generator.
    *   **Metrics**: Displays total packets, anomaly count, and system status (Normal/Critical).
    *   **Charts**: Live line charts for traffic volume and anomaly detections.

---

## 4. Workflows

### 4.1 Training Workflow (`main.py`)
1.  **Load Data**: Reads the UNSW-NB15 testing set.
2.  **Fit Preprocessor**: Learns scaling parameters and categorical mappings.
3.  **Train Models**: 
    *   Isolation Forest fits on full training batch.
    *   Autoencoder fits primarily on 'Normal' traffic (label=0) to learn baseline behavior.
4.  **Save Artifacts**: Serializes all components to disk for the dashboard to load.

### 4.2 Monitoring Workflow
1.  **Load Models**: Dashboard loads `preprocessor.pkl` and model files on startup.
2.  **Stream**: User clicks "Start Monitoring".
3.  **Inference Loop**:
    *   Batch $N$ arrives.
    *   Preprocessor transforms $N$.
    *   Models predict independently.
    *   **Ensemble Logic**: Anomaly if *any* model flags the packet (OR gate) or *all* models flag it (AND gate), depending on configuration.
4.  **Alert**: UI updates with red/green status indicators.

---

## 5. Deployment & Testing

### 5.1 Requirements
*   Python 3.8+
*   TensorFlow 2.x
*   Streamlit
*   Pandas/NumPy/Scikit-learn

### 5.2 Running Unit Tests
The system includes a `unittest` suite in `tests/`.
```bash
python tests/test_components.py
```

### 5.3 Quick Start
```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Train models (Initial setup)
python main.py

# 3. Launch Dashboard
streamlit run src/dashboard/app.py
```

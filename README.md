#  Project Overview
This project focuses on Anomaly Detection in the Tennessee Eastman Process (TEP), a well-known industrial benchmark used for process control and fault detection research.

We implement multiple machine learning and deep learning models to detect anomalies in TEP sensor data, with options for fast approximate detection or accurate but computationally heavier detection.

The pipeline is modular and extensible, allowing you to:

   - Run anomaly detection using Isolation Forest, Local Outlier Factor (LOF), One-Class SVM, PCA, and Autoencoder (TensorFlow/Keras).
    
   - Select between Fast Mode (optimized for speed, still preserves feature importance length) and Accurate Mode (full evaluation across models).
    
   - Extract Top contributing features per data entry, explaining anomalies.
    
   - Save and visualize model results for further analysis.

# Installation & Setup
1️. Clone the Repository
```
git clone https://github.com/TeresaSaraSaji/AnomalySense.git
```
```
cd AnomalySense
```

2️. Create Virtual Environment
```
python -m venv venv
```
Linux / macOS
```
source venv/bin/activate
```
Windows PowerShell
```
venv\Scripts\activate
```
3️. Install Dependencies
```
pip install -r web/requirements.txt
```
(for AutoEncoders)
```
pip install tensorflow
```
# Usage
The main script is models/run_algorithms.py.
It supports:

Modes: fast (speed) or accurate (full evaluation).

Models: choose one (isoforest, lof, svm, autoencoder, pca) or all.

Run:
```
python web/app.py
```

After that select:

    1. File
    2. Model
    3. Mode

    
Output:
   - Top 7 anomaly features per entry as columns.
   - Anomaly scores + predictions saved in results.
   - This file can be downloaded.

# Models Implemented
 - Isolation Forest (Sklearn)

 - Local Outlier Factor (LOF)

 - KNN

 - Mahalanobis

 - One-Class SVM

 - Principal Component Analysis (PCA-based anomaly detection)

 - Deep Autoencoder (TensorFlow/Keras)

# Configuration
You can configure:

Dataset path (data/ directory)

Model parameters (edit config.json or modify arguments)

Output - Downloadable sheet

# Example Workflow
Prepare the dataset in data/.

Run anomaly detection:
```
python web/app.py
```
Download the sheet from the given option.

anomaly_scores.csv → anomaly scores per sample

top_features.csv → Top contributing features per entry




# AI-Driven Log Anomaly Detector

An autonomous, unsupervised Machine Learning application designed to analyze system logs and detect anomalies (e.g., security breaches, SQL injections, or system failures) without the need for predefined rules or labeled training data.

## Features
- **NLP Preprocessing:** Uses TF-IDF vectorization to convert raw text logs into numerical features.
- **Unsupervised ML:** Leverages the `Isolation Forest` algorithm to isolate rare data points and structural outliers.
- **Modular Design:** Clean separation of concerns between data preprocessing, model execution, and entry points.

## Tech Stack
- Python 3.x
- Scikit-Learn
- Joblib

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline:
   ```bash
   cd src
   python main.py
   ```

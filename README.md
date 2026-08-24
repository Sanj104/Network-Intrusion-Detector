#  Network Intrusion Detector

A machine learning-based web application that detects whether network traffic is **Normal** or a potential **Attack** using the NSL-KDD dataset.

## Features

- Network traffic analysis
- Multiple ML model comparison
- Automatic best-model selection
- Accuracy, Precision, Recall and F1-score
- Confusion Matrix and ROC Curve
- Interactive Flask web dashboard
- Real-time traffic prediction and risk classification

## Tech Stack

- Python
- Scikit-learn
- Pandas & NumPy
- Flask
- Matplotlib & Seaborn
- HTML & CSS

## Models

- Logistic Regression
- Decision Tree
- Random Forest

**Best Model:** Decision Tree — 78.49% accuracy

## Run Locally

```bash
pip install -r requirements.txt
python app.py

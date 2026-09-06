# Heart Disease Prediction

A machine learning project that predicts the likelihood
of heart disease using a KNN classification model.

## Live Demo

https://heart-disease-prediction-ml-2106.streamlit.app

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Project Workflow

1. Data preprocessing
2. Exploratory Data Analysis
3. Feature scaling
4. KNN model training
5. Model evaluation
6. Streamlit application

## Model Performance

Several classification algorithms were trained and evaluated
to identify the best-performing model.

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 87.50% | 88.78% |
| **KNN** | **88.59%** | **89.86%** |
| Naive Bayes | 86.96% | 87.88% |
| Decision Tree | 76.63% | 77.72% |
| SVM | 86.41% | 88.04% |

### Best Model

The **K-Nearest Neighbors (KNN)** model achieved the highest
performance among the tested models, with:

- **Accuracy:** 88.59%
- **F1 Score:** 89.86%

Therefore, KNN was selected for the final Streamlit application.

## Model

K-Nearest Neighbors (KNN)


## Disclaimer

This project is for educational purposes only and is
not intended for medical diagnosis.

## Project Structure


```text
Heart-Disease-Prediction/
│
├── data/
│   └── heart.csv
│
├── app.py
├── HeartDisease.ipynb
├── KNN_heart_model.pkl
├── heart_scaler.pkl
├── heart_columns.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

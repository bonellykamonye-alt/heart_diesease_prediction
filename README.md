Heart Disease Prediction using Supervised Machine Learning

A machine learning project that predicts the presence of heart disease in patients based on clinical data, built as part of my ongoing portfolio of supervised learning projects while studying towards a BSc in Business Information Technology at Dedan Kimathi University of Technology.

Live App: https://bonellykamonye-alt-heart-diesease-prediction-app-l25pjl.streamlit.app

1. Introduction

Cardiovascular disease remains one of the leading causes of death worldwide. Early detection plays a critical role in enabling timely medical intervention and improving patient outcomes. This project explores how machine learning can be applied to clinical data to predict the likelihood of heart disease in a patient, using a dataset of real (anonymized) patient records.

The objective of this project was twofold: first, to practice the full supervised learning workflow — from data exploration through to model deployment — and second, to build a practical, interactive tool that demonstrates these skills to a wider audience.

Disclaimer: This project was built for academic and portfolio purposes only. It is not intended for real-world medical diagnosis and should not be used as a substitute for professional healthcare advice.

2. Dataset

The dataset used is the Heart Disease Dataset from the UCI Machine Learning Repository (Cleveland subset), a widely used benchmark dataset in the machine learning community.

Source: UCI Machine Learning Repository
Observations: 303 patient records
Features: 13 clinical attributes
Target variable: target (1 = presence of heart disease, 0 = absence)
Feature Description
Column	Description
age	Age of the patient in years
sex	Sex (1 = male, 0 = female)
cp	Chest pain type (0–3)
trestbps	Resting blood pressure (mm Hg)
chol	Serum cholesterol (mg/dl)
fbs	Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
restecg	Resting electrocardiographic results (0–2)
thalach	Maximum heart rate achieved
exang	Exercise-induced angina (1 = yes, 0 = no)
oldpeak	ST depression induced by exercise relative to rest
slope	Slope of the peak exercise ST segment (0–2)
ca	Number of major vessels colored by fluoroscopy (0–3)
thal	Thalassemia type (1 = normal, 2 = fixed defect, 3 = reversible defect)
target	Diagnosis of heart disease (1 = disease, 0 = no disease)
3. Exploratory Data Analysis

Before modelling, I conducted exploratory data analysis to understand the underlying structure of the data and its relationship with the target variable. Key observations included:

Age distributions between patients with and without heart disease overlapped considerably, suggesting age alone is not a strong standalone predictor.
Cholesterol levels followed a similar pattern, with substantial overlap between the two groups and a small number of high outliers.
Features such as chest pain type, maximum heart rate, and ST depression showed comparatively clearer separation between the two classes, indicating stronger predictive potential.
4. Data Preprocessing

To prepare the data for modelling, the following steps were applied:

Continuous features (age, trestbps, chol, thalach, oldpeak, ca) were standardized using StandardScaler.
Categorical features (cp, restecg, thal, slope) were one-hot encoded, as they represent nominal categories rather than ordered numerical values.
Binary features (sex, fbs, exang) required no transformation.
The dataset was split into training (80%) and testing (20%) sets, stratified on the target variable to preserve class balance.
5. Model Development and Comparison

Four candidate models were trained and evaluated on the same preprocessed dataset to compare performance:

Model	Accuracy	Recall (Disease)	Precision (Disease)	ROC-AUC
Logistic Regression	0.80	0.85	0.80	0.799
Random Forest	0.77	0.79	0.79	0.845
Support Vector Machine (RBF)	0.75	0.79	0.76	0.846
Gradient Boosting	0.72	0.73	0.75	0.817

Given the healthcare context of this problem, recall was prioritized over raw accuracy, as failing to identify a true positive case (a patient who does have heart disease) carries greater consequence than a false alarm.

6. Hyperparameter Tuning

The SVM model was selected for further tuning using GridSearchCV, testing combinations of the C and gamma parameters.

An initial tuning attempt using recall as the sole scoring metric produced a model that achieved perfect recall on the disease class but performed poorly on the healthy class — a clear sign of an imbalanced, impractical model. The scoring metric was subsequently changed to f1, which balances precision and recall, resulting in a more reliable and generalizable model.

Best parameters identified: C = 1, gamma = 0.01

7. Final Model Performance

The final selected model is a Support Vector Machine (RBF kernel), tuned via GridSearchCV using F1 score.

Metric	Score
Accuracy	0.80
Recall (Disease)	0.88
Precision (Disease)	0.78
ROC-AUC	0.876

This model was chosen as the final model because it achieved the strongest ROC-AUC score across all candidates while maintaining accuracy and recall comparable to, or better than, the other models tested — without requiring manual adjustment of the classification threshold.

8. Technology Stack
Language: Python
Data Handling & Analysis: pandas, numpy
Visualization: matplotlib
Modelling: scikit-learn
Model Persistence: joblib
Deployment: Streamlit
Development Environment: Jupyter Notebook
9. Project Structure
heart_diesease_prediction/
├── data/
│   └── heart_disease_dataset.csv
├── notebooks/
│   └── heart_disease.ipynb
├── heart_disease_model.pkl
├── heart_disease_preprocessor.pkl
├── app.py
├── requirements.txt
└── README.md
10. How to Run This Project Locally
bash
git clone https://github.com/bonellykamonye-alt/heart_diesease_prediction.git
cd heart_diesease_prediction
pip install -r requirements.txt
streamlit run app.py
11. Author

Bonelly Kamonye BSc Business Information Technology Dedan Kimathi University of Technology, Kenya

This project is part of an ongoing personal portfolio of supervised machine learning projects, developed to build practical experience in the end-to-end data science workflow — from data exploration to model deployment.
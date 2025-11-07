# Churn-Prediction-using-Machine-Learning
# AI-Powered Customer Retention Prediction System

![Churn Prediction](https://img.shields.io/badge/Project-Customer%20Churn-blue)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Machine Learning Models](#machine-learning-models)
6. [Features](#features)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Results](#results)
10. [Future Enhancements](#future-enhancements)
11. [Contributors](#contributors)

---

## Project Overview
Customer retention is a key challenge for businesses, as acquiring new customers is costlier than retaining existing ones. This project builds an **AI-powered system** to predict customers likely to churn and provide actionable insights for proactive retention strategies. The system uses machine learning to forecast churn probability based on customer demographics, usage behavior, billing patterns, and support interactions.

---

## Business Problem
- Identify customers at risk of leaving the service (churners)
- Reduce revenue loss due to unexpected customer exits
- Enable proactive engagement strategies to improve retention
- Segment customers based on risk and loyalty for personalized actions

---

## Dataset
The dataset is sourced from Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Key Columns:**
| Column Name       | Data Type | Description |
|------------------|-----------|-------------|
| customerID        | string    | Unique customer identifier |
| gender            | string    | Male/Female |
| SeniorCitizen     | int       | 0 = No, 1 = Yes |
| Partner           | string    | Yes/No |
| Dependents        | string    | Yes/No |
| tenure            | int       | Months with company |
| PhoneService      | string    | Yes/No |
| MultipleLines     | string    | Yes/No/No phone service |
| InternetService   | string    | DSL/Fiber optic/No |
| OnlineSecurity    | string    | Yes/No/No internet service |
| Contract          | string    | Month-to-month/One year/Two year |
| PaperlessBilling  | string    | Yes/No |
| PaymentMethod     | string    | Payment type |
| MonthlyCharges    | float     | Monthly billing amount |
| TotalCharges      | float     | Total charges |
| Churn             | string    | Target variable: Yes = churn, No = retained |

---

## Methodology
1. **Data Preprocessing**
   - Handle missing values
   - Outlier treatment using **IQR method**
   - Encode categorical features (OneHot, Ordinal, Label Encoding)
   - Feature scaling (StandardScaler)
   - Balance dataset using **SMOTE**

2. **Feature Selection**
   - Chi-Square test for categorical features
   - ANOVA, t-test, and correlation for numerical features

3. **Model Development**
   - Train multiple models
   - Evaluate using Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Select best performing model (Gradient Boosting/XGBoost)

4. **Deployment**
   - User-friendly interface using Flask/Streamlit
   - Input new customer data to get churn predictions

---

## Machine Learning Models
- **Logistic Regression** – Baseline classifier
- **K-Nearest Neighbors (KNN)** – Distance-based classification
- **Naive Bayes** – Probabilistic classifier
- **Decision Tree** – Tree-based model using entropy
- **Random Forest** – Ensemble of decision trees
- **XGBoost (Gradient Boosting)** – Advanced boosting algorithm, selected as final model
- **Support Vector Machine (SVM)** – For linear/non-linear separation

*Note:* All models were trained with default hyperparameters, with XGBoost selected due to its higher accuracy and ROC-AUC performance.

---

## Features
- Customer demographics
- Tenure and contract type
- Payment method and billing patterns
- Internet and phone services
- Churn probability prediction

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Preprocess data and train models:

bash
Copy code
python train_model.py
Run the Flask app for prediction:

bash
Copy code
python app.py
Open http://127.0.0.1:5000/ in your browser to interact with the prediction system.

Results
XGBoost achieved 80% accuracy on the test set.

AUC-ROC curves generated to compare model performances.

Confusion matrices and classification reports provide detailed model insights.

Future Enhancements
Integrate sentiment analysis from customer feedback

Deploy into CRM systems for real-time monitoring

Explore Deep Learning (RNN/LSTM) for temporal churn patterns

Implement automated retention campaigns triggered by churn probability


📊 AI-Powered Customer Retention Prediction System
Churn Prediction using Machine Learning
🧠 Project Overview

Customer retention is one of the biggest challenges faced by telecom and service-based businesses. Acquiring new customers costs significantly more than retaining existing ones.
This AI-powered Churn Prediction System helps businesses identify customers who are likely to discontinue their services and enables data-driven retention strategies.

The system leverages Machine Learning algorithms to predict customer churn based on demographics, service usage, billing behavior, and contract information.

🚨 Business Problem

Companies lose millions due to unexpected customer churn.
This project addresses the following challenges:

Detect customers likely to churn before they leave

Reduce revenue loss and acquisition costs

Understand key churn factors for business insights

Enable proactive engagement and personalized retention offers

🗂️ Dataset

Dataset: Telco Customer Churn – Kaggle

Key Features
Feature	Description
customerID	Unique ID for each customer
gender	Male / Female
SeniorCitizen	1 = Yes, 0 = No
Partner, Dependents	Relationship-based attributes influencing loyalty
tenure	Number of months the customer has been with the company
PhoneService, MultipleLines	Telephone service details
InternetService, OnlineBackup, TechSupport	Type and usage of internet-related services
Contract	Month-to-month / One-year / Two-year
PaperlessBilling	Yes / No
PaymentMethod	Electronic check / Bank transfer / Credit card / Mailed check
MonthlyCharges, TotalCharges	Customer billing information
Churn	Target variable — Yes (churned) / No (retained)
⚙️ Methodology
1️⃣ Data Preprocessing

Handled missing values and inconsistent data

Detected and capped outliers using the IQR (Interquartile Range) method

Encoded categorical features using OneHot, Ordinal, and Label Encoding

Scaled numerical columns with StandardScaler

Balanced imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique)

2️⃣ Feature Selection

Chi-Square Test → for categorical features

ANOVA, t-Test, Correlation → for numerical features

3️⃣ Model Training and Evaluation

Trained multiple models using scikit-learn and XGBoost, comparing their performance using:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Curve

4️⃣ Deployment

A Flask Web Application was built to:

Take new customer inputs through an interactive UI

Predict churn probability

Display prediction confidence and category

🤖 Machine Learning Models Evaluated
Model	Description
Logistic Regression	Baseline linear classifier
K-Nearest Neighbors (KNN)	Instance-based model using distance metrics
Naive Bayes	Probabilistic model assuming feature independence
Decision Tree	Tree-based structure using entropy/gini
Random Forest	Ensemble of decision trees to reduce variance
Support Vector Machine (SVM)	Linear/non-linear classifier using kernels
Gradient Boosting (XGBoost)	Combines weak learners; handles complex data efficiently ✅ (Final Model Selected)

Model Finalization:
After studying various ML algorithms and evaluating their results, Gradient Boosting Classifier was finalized for deployment due to its superior performance, ability to handle non-linearity, and robustness against overfitting.
The model was trained using default parameters and achieved high stability and accuracy across datasets.

🧾 Features of the Final System

Interactive web interface using Flask

Dynamic encoding and feature alignment

Scaled and standardized numeric fields

Real-time churn prediction with confidence score


📈 Results & Evaluation

Final Model: Gradient Boosting Classifier

Test Accuracy: ~80%

Precision/Recall Balance: Excellent

AUC-ROC Curve: Demonstrated clear class separability

Confusion Matrix: Validated consistent predictions across churn/non-churn categories

🚀 Future Enhancements

Integrate Sentiment Analysis from customer feedback or social media

Deploy into CRM systems for real-time retention tracking

Use Deep Learning (LSTM/RNN) to model time-based churn patterns

Build automated retention campaign triggers based on churn probability

👩‍💻 Developer

Name: K. Yamini Krishna
Email: yamini.kelam1@gmail.com

Phone: +91 81218 54228
GitHub: github.com/yaminikrish

LinkedIn: linkedin.com/in/yamini-krishna-445412367

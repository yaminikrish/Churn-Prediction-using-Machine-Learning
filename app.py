from flask import Flask, render_template, request
import pandas as pd
import pickle
import traceback

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("gradient_boosting.pkl", "rb"))
scaler = pickle.load(open("gb.pkl", "rb"))
final_cols = model.feature_names_in_

# Developer info
developer_info = {
    "name": "Your Name",
    "phone": "+91-XXXXXXXXXX",
    "email": "youremail@gmail.com",
    "github": "https://github.com/yourusername",
    "linkedin": "https://linkedin.com/in/yourusername"
}

# Model info
model_info = {
    "model_name": "Gradient Boosting Classifier",
    "description": "Predicts if a customer is likely to churn based on their demographics, services, and billing patterns.",
    "columns": [
        {"name": "gender", "desc": "Customer gender: Male/Female"},
        {"name": "SeniorCitizen", "desc": "0 = No, 1 = Yes"},
        {"name": "Partner", "desc": "Whether customer has a partner: Yes/No"},
        {"name": "Dependents", "desc": "Whether customer has dependents: Yes/No"},
        {"name": "tenure", "desc": "Number of months customer has stayed"},
        {"name": "PhoneService", "desc": "Phone service subscription: Yes/No"},
        {"name": "MultipleLines", "desc": "Multiple phone lines: Yes/No/No phone service"},
        {"name": "InternetService", "desc": "DSL/Fiber optic/No"},
        {"name": "OnlineSecurity", "desc": "Online security subscription: Yes/No/No internet service"},
        {"name": "OnlineBackup", "desc": "Online backup subscription: Yes/No/No internet service"},
        {"name": "DeviceProtection", "desc": "Device protection subscription: Yes/No/No internet service"},
        {"name": "TechSupport", "desc": "Tech support subscription: Yes/No/No internet service"},
        {"name": "StreamingTV", "desc": "Streaming TV subscription: Yes/No/No internet service"},
        {"name": "StreamingMovies", "desc": "Streaming movies subscription: Yes/No/No internet service"},
        {"name": "Contract", "desc": "Contract type: Month-to-month/One year/Two year"},
        {"name": "PaperlessBilling", "desc": "Paperless billing: Yes/No"},
        {"name": "PaymentMethod", "desc": "Payment method: Electronic check / Mailed check / Credit card / Bank transfer"},
        {"name": "MonthlyCharges_qan_iqr", "desc": "Monthly charges after preprocessing (numeric)"},
        {"name": "TotalCharges_itr_qan_iqr", "desc": "Total charges after preprocessing (numeric)"}
    ]
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data as DataFrame
        data = {k: v for k, v in request.form.items()}
        df = pd.DataFrame([data])

        # --- SIM Encoding ---
        if 'SIM' in df.columns:
            sim_map = {'jio': 0, 'airtel': 1, 'bsnl': 2, 'vi': 3}
            df['SIM'] = df['SIM'].astype(str).str.lower().map(sim_map).fillna(0)
        else:
            df['SIM'] = 0

        # --- Categorical Encoding ---
        categorical_mappings = {
            'Gender': {'Male': 0, 'Female': 1},
            'Partner': {'No': 0, 'Yes': 1},
            'Dependents': {'No': 0, 'Yes': 1},
            'PhoneService': {'No': 0, 'Yes': 1},
            'MultipleLines': {'No': 0, 'Yes': 1, 'No phone service': 0},
            'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
            'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 0},
            'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 0},
            'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 0},
            'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 0},
            'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 0},
            'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 0},
            'PaperlessBilling': {'No': 0, 'Yes': 1},
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'PaymentMethod': {
                'Electronic check': 0,
                'Mailed check': 1,
                'Credit card (automatic)': 2,
                'Bank transfer (automatic)': 3
            }
        }

        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)

        # --- Numeric Columns ---
        numeric_cols = ['MonthlyCharges_qan_iqr', 'TotalCharges_itr_qan_iqr']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0

        # --- Apply scaling ---
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # --- Align columns to match training order ---
        for col in final_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[final_cols]

        # --- Predict ---
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1] * 100

        result = "🚨 Customer Likely to CHURN" if pred == 1 else "✅ Customer Will STAY"
        return render_template("result.html", prediction_text=result, proba=round(proba, 2))

    except Exception as e:
        print("⚠️ Error:", e)
        traceback.print_exc()
        return f"⚠️ Error during prediction: {e}"

@app.route('/about_developer')
def about_developer():
    return render_template("about_developer.html", developer=developer_info)

@app.route('/about_model')
def about_model():
    return render_template("about_model.html", model=model_info)

if __name__ == "__main__":
    app.run(debug=True)

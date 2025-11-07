from flask import Flask, render_template, request
import pandas as pd
import pickle
import traceback

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("gradient_boosting.pkl", "rb"))
scaler = pickle.load(open("gb.pkl", "rb"))
final_cols = model.feature_names_in_

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

        # --- Numeric Columns (exact names from model) ---
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

if __name__ == "__main__":
    app.run(debug=True)

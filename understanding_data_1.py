import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from log_code import setup_logging
logger=setup_logging('understanding_data1')
logger.info('logging successfull')

import sys

class TelcoChurnEDA:
    def __init__(self, filepath):
        """Initialize dataset and basic setup"""
        try:
            self.df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
            logger.info('Dataset loaded successfully')
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def basic_info(self):
        """Display dataset structure and summary"""
        try:
            print("\n--- Dataset Info ---")
            print(self.df.shape)
            print(self.df.columns)
            print(self.df.info())
        except Exception as e:
            print(f"Error in basic_info: {e}")

    def plot_churn_distribution(self):
        """Pie chart showing churned vs non-churned customers"""
        try:
            plt.figure(figsize=(5, 3))
            counts = self.df['Churn'].value_counts()
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
            plt.title('Churn Distribution')
            plt.show()
        except Exception as e:
            print(f"Error in plot_churn_distribution: {e}")

    def senior_vs_churn(self):
        """Bar chart comparing churn rate of senior citizens"""
        try:
            plt.figure(figsize=(6, 4))
            sns.countplot(x='SeniorCitizen', hue='Churn', data=self.df, palette=['green', 'red'])
            plt.title('Senior Citizen vs Churn')
            plt.show()
        except Exception as e:
            print(f"Error in senior_vs_churn: {e}")

    def numeric_distribution(self):
        """Histograms for numerical columns categorized by churn"""
        try:
            numerical_cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges'] if c in self.df.columns]
            if not numerical_cols:
                print("No numerical columns found for plotting.")
                return

            plt.figure(figsize=(15, 4))
            for i, col in enumerate(numerical_cols, 1):
                plt.subplot(1, len(numerical_cols), i)
                plt.hist(self.df[self.df['Churn'] == 'No'][col].dropna(), bins=10, label='No Churn', color='green', alpha=0.7)
                plt.hist(self.df[self.df['Churn'] == 'Yes'][col].dropna(), bins=10, label='Churned', color='red', alpha=0.7)
                plt.title(f'{col} Distribution by Churn')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in numeric_distribution: {e}")

    def contract_analysis(self):
        """Visualize contract type distribution and churn impact"""
        try:
            if 'Contract' not in self.df.columns:
                print("'Contract' column not found.")
                return
            contract_churn = self.df.groupby(['Contract', 'Churn']).size().unstack(fill_value=0)
            contract_churn.plot(kind='bar', stacked=True, color=['green', 'red'])
            plt.title('Contract Type vs Churn')
            plt.xlabel('Contract Type')
            plt.ylabel('Number of Customers')
            plt.legend(title='Churn')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in contract_analysis: {e}")

    def categorical_churn_analysis(self):
        """Visualize all categorical columns vs churn"""
        try:
            categorical_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
            for col in categorical_cols:
                if col != 'Churn':
                    plt.figure(figsize=(7, 5))
                    sns.countplot(data=self.df, x=col, hue='Churn', palette=['green', 'red'])
                    plt.title(f'{col} vs Churn')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.show()
        except Exception as e:
            print(f"Error in categorical_churn_analysis: {e}")

    def churn_trend_by_tenure(self):
        """Area chart showing trend of average monthly charges by tenure and churn"""
        try:
            if 'tenure' not in self.df.columns or 'MonthlyCharges' not in self.df.columns:
                print("'tenure' or 'MonthlyCharges' column not found.")
                return
            monthly_churn = self.df.groupby(['tenure', 'Churn'])['MonthlyCharges'].mean().reset_index()
            plt.figure(figsize=(8, 4))
            for churn_status in monthly_churn['Churn'].unique():
                temp = monthly_churn[monthly_churn['Churn'] == churn_status]
                plt.fill_between(temp['tenure'], temp['MonthlyCharges'], alpha=0.5, label=f"Churn: {churn_status}")
                plt.plot(temp['tenure'], temp['MonthlyCharges'], linewidth=2)
            plt.title("Monthly Charges Trend by Churn Over Tenure")
            plt.xlabel("Tenure (Months)")
            plt.ylabel("Average Monthly Charges ($)")
            plt.legend(title="Churn Status")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in churn_trend_by_tenure: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    try:
        churn_eda = TelcoChurnEDA(r"C:\Users\Dell\Downloads\churnProject\WA_Fn-UseC_-Telco-Customer-Churn.csv")
        churn_eda.basic_info()
        churn_eda.plot_churn_distribution()
        churn_eda.senior_vs_churn()
        churn_eda.numeric_distribution()
        churn_eda.contract_analysis()
        churn_eda.categorical_churn_analysis()
        churn_eda.churn_trend_by_tenure()
        logger.info("All analysis completed successfully.")
    except Exception as e:
        logger.info(f"Execution failed due to: {e}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from log_code import setup_logging
from feature_engine.outliers import Winsorizer
logger=setup_logging('handling_outlier')
import sklearn
from scipy import stats



def trim_outliers(x_train,x_test):
    try:
        df = x_train.copy()
        df1=x_test.copy()
        logger.info(df.columns)
        logger.info(f'check_columns')
        cols = ['MonthlyCharges_qan', 'TotalCharges_itr_qan']


        for i in cols:
            Q1 = df[i].quantile(0.25)
            Q3 = df[i].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[i+'_trim']=np.where(df[i]>upper,upper,np.where(df[i]<lower,lower,df[i]))
            df1[i + '_trim'] = np.where(df1[i] > upper, upper, np.where(df1[i] < lower, lower, df1[i]))
      # After-trimming boxplot
            plt.figure(figsize=(8, 3))
            sns.boxplot(x=df[i+'_trim'], color='lightgreen')
            plt.title(f'{i+'_trim'} - After Trimming')
            plt.show()

        logger.info(f"Trimming completed. Shape after trimming: {df.shape}")
        logger.info(f"Trimming completed. Shape after trimming: {df1.shape}")
        return df,df1
    except Exception as e:
        import sys
        e_type, e_value, e_tb = sys.exc_info()
        logger.error(f"Issue in dividing() at line {e_tb.tb_lineno}: {e_value}")

def caping(x_train, x_test, method, fold):
    try:
        df = x_train.copy()
        df1 = x_test.copy()
        cols = ['MonthlyCharges_qan', 'TotalCharges_itr_qan']
        winsor = Winsorizer(capping_method=method, tail='both', fold=fold, variables=cols)
        df_trans = winsor.fit_transform(df)
        df1_trans = winsor.transform(df1)
        for i in cols:
            new_col_train = f"{i}_{method}"
            new_col_test = f"{i}_{method}"
            df[new_col_train] = df_trans[i]
            df1[new_col_test] = df1_trans[i]
            plt.figure(figsize=(12, 3))
            plt.subplot(1, 3, 1)
            sns.boxplot(x=df[new_col_train], color='r')
            plt.title(f'{new_col_train} - After {method}')

            plt.subplot(1, 3, 2)
            df[new_col_train].plot(kind='kde', color='skyblue')
            plt.title(f'{new_col_train} - Bell {method} ')
            plt.xlabel(new_col_train)
            plt.ylabel('Density')

            plt.subplot(1, 3, 3)
            stats.probplot(df[new_col_train], dist="norm", plot=plt)
            plt.title(f'{new_col_train} - Q-Q Plot')
            plt.show()
        logger.info(f"Capping using {method} Winsorizer completed successfully.")
        return df, df1
    except Exception:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f"Issue is at line {e_linno.tb_lineno} due to {e_msg}")






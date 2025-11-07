import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from scipy.special import lambertw
from log_code import setup_logging
logger=setup_logging('trans')
import sklearn
import sys
from scipy import stats

def log(x_train):
    try:
        df = x_train.copy()
        for i in df.columns:
            new_col_name = f"{i}_log"
            df[new_col_name] = np.log(df[i] + 1)
        return df
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

def square_root(x_train):
    try:
        df = x_train.copy()
        logger.info(df.columns)
        for i in df.columns:
            new_col_name=f'{i}_sqrt'
            df[new_col_name]=df[i]**0.2
        return df
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

def reci(x_train):
    try:
        df = x_train.copy()
        logger.info(df.columns)
        for i in df.columns:
            new_col_name = f'{i}_rec'
            df[new_col_name] = 1.0/(df[i])
        return df
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

def cbrt(x_train):
    try:
        df = x_train.copy()
        logger.info(df.columns)
        for i in df.columns:
            new_col_name = f'{i}_cb'
            df[new_col_name] = np.cbrt(df[i])
        return df
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

def quantile(x_train,x_test):
    try:
        df = x_train.copy()
        df1=x_test.copy()
        qt = QuantileTransformer(output_distribution='normal', random_state=24)
        transformed = qt.fit_transform(df)
        transformed1 = qt.fit_transform(df1)
        a=pd.DataFrame(transformed,columns=[i + '_qan' for i in df.columns],index=df.index)
        b= pd.DataFrame(transformed1, columns=[i + '_qan' for i in df1.columns], index=df1.index)
        return a,b
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

def hyper_arsine(x_train):
    try:
        df = x_train.copy()
        logger.info(df.columns)
        for i in df.columns:
            new_col_name = f'{i}_hyp'
            df[new_col_name] = np.arcsinh(df[i])
        return df
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')


def rank_quantile(x_train):
    try:
        df = x_train.copy()
        qt = QuantileTransformer(output_distribution='normal', random_state=24)
        transformed = qt.fit_transform(df)
        a=pd.DataFrame(transformed,columns=[i + '_rank' for i in df.columns],index=df.index)
        return a
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

def powertrans(x_train):
    try:
        df = x_train.copy()
        pt=PowerTransformer(method='yeo-johnson',standardize=True)
        for i in df.columns:
            new_col_name = f'{i}_yeo'
            df[new_col_name] = pt.fit_transform(df[[i]])
        return df
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')
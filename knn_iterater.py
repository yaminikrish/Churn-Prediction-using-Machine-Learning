import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import logging
import os
import sys
from log_code import setup_logging
logger = setup_logging('iterative_imputer')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor

def iteration_decision_tree(x_train, x_test):
    try:
        total_charges = x_train['TotalCharges'].copy()
        total_char = x_test['TotalCharges'].copy()
        cont_cols = ['MonthlyCharges', 'TotalCharges']
        x_train_cont = x_train[cont_cols].copy()
        x_test_cont = x_test[cont_cols].copy()
        imp = IterativeImputer(estimator=DecisionTreeRegressor(max_depth=5, random_state=42),max_iter=10,random_state=42)
        x_train_imp = imp.fit_transform(x_train_cont)
        x_train_cont_imp = pd.DataFrame(x_train_imp, columns=cont_cols, index=x_train.index)
        x_test_imp = imp.transform(x_test_cont)
        x_test_cont_imp = pd.DataFrame(x_test_imp, columns=cont_cols, index=x_test.index)
        x_train['TotalCharges_itr'] = x_train_cont_imp['TotalCharges']
        x_test['TotalCharges_itr'] = x_test_cont_imp['TotalCharges']
        std_before = total_charges.std()
        std_after = x_train['TotalCharges_itr'].std()
        logger.info(f"Standard Deviation BEFORE imputation (TotalCharges): {std_before:.4f}")
        logger.info(f"Standard Deviation AFTER imputation (TotalCharges): {std_after:.4f}")
        logger.info(x_train.info())
        logger.info(x_test.info())
        return x_train, x_test

    except Exception as e:
        e_type, e_msg, e_tb = sys.exc_info()
        logger.info(f"Issue is: {e_tb.tb_lineno} due to {e_msg}")
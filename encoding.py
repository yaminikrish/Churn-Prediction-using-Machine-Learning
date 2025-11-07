import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from log_code import setup_logging
logger=setup_logging('encoding file')

from sklearn.preprocessing import OneHotEncoder





def cattonum(x_train_cat,x_test_cat):
    try:
        cols=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','PaymentMethod','sim']
        oh = OneHotEncoder(categories='auto', drop='first', handle_unknown='ignore')
        oh.fit(x_train_cat[cols])
        logger.info(f'{oh.categories_}')
        logger.info(f'{oh.get_feature_names_out()}')
        res_train = oh.transform(x_train_cat[cols]).toarray()
        res_test = oh.transform(x_test_cat[cols]).toarray()
        f_train = pd.DataFrame(res_train, columns=oh.get_feature_names_out())
        f_test = pd.DataFrame(res_test, columns=oh.get_feature_names_out())
        x_train_cat.reset_index(drop=True, inplace=True)
        f_train.reset_index(drop=True, inplace=True)
        x_test_cat.reset_index(drop=True, inplace=True)
        f_test.reset_index(drop=True, inplace=True)
        # Concatenate encoded columns with original data
        x_train_cat = pd.concat([x_train_cat.drop(columns=cols), f_train], axis=1)
        x_test_cat = pd.concat([x_test_cat.drop(columns=cols), f_test], axis=1)

        logger.info(f'{x_train_cat.isnull().sum()}')
        logger.info(f'{x_test_cat.isnull().sum()}')

        logger.info(f'{x_train_cat.sample(10)}')
        logger.info(f'{x_test_cat.sample(10)}')

        return x_train_cat,x_test_cat

    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f' Issue is at line {e_linno.tb_lineno} due to {e_msg}')
import pandas as pd
import numpy as np
import logging
import sys
from log_code import setup_logging
logger = setup_logging('random_sample_imputer')

def random_sample_imputation(x_train, x_test, column):
    try:
        logger.info(f'Before Random Sample Imputation')
        logger.info(f'Train {column} missing: {x_train[column].isnull().sum()}')
        logger.info(f'Test {column} missing: {x_test[column].isnull().sum()}')

        # Keep original copy for comparison
        before_imp_train = x_train[column].copy()

        # --- Create copies for random-sample results ---
        x_train[f'{column}_rand'] = x_train[column].copy()
        x_test[f'{column}_rand'] = x_test[column].copy()

        # --- Random sample imputation on TRAIN ---
        if x_train[column].isnull().sum() > 0:
            train_random = (
                x_train[column]
                .dropna()
                .sample(x_train[column].isnull().sum(), replace=True, random_state=24)
            )
            train_random.index = x_train[x_train[column].isnull()].index
            x_train.loc[x_train[column].isnull(), f'{column}_rand'] = train_random

        # --- Random sample imputation on TEST ---
        if x_test[column].isnull().sum() > 0:
            source = x_test[column].dropna()
            if source.empty:
                source = x_train[column].dropna()
            test_random = source.sample(
                x_test[column].isnull().sum(), replace=True, random_state=24
            )
            test_random.index = x_test[x_test[column].isnull()].index
            x_test.loc[x_test[column].isnull(), f'{column}_rand'] = test_random

        # --- Logging post-imputation ---
        logger.info(f'After Random Sample Imputation')
        logger.info(f'Train {column} missing: {x_train[f"{column}_rand"].isnull().sum()}')
        logger.info(f'Test {column} missing: {x_test[f"{column}_rand"].isnull().sum()}')

        # --- Standard deviation comparison ---
        std_before = before_imp_train.std(skipna=True)
        std_after = x_train[f'{column}_rand'].std(skipna=True)
        logger.info(f"Standard Deviation BEFORE: {std_before:.4f}")
        logger.info(f"Standard Deviation AFTER: {std_after:.4f}")

        # --- DataFrame info logs ---
        logger.info(x_train.info())
        logger.info(x_test.info())

        return x_train, x_test

    except Exception as e:
        e_type, e_msg, e_tb = sys.exc_info()
        logger.error(f"Issue in line {e_tb.tb_lineno} due to {e_msg}")
        return x_train, x_test
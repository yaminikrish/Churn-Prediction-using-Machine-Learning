import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger=setup_logging('filter_methods')
from sklearn.feature_selection import chi2, f_classif, f_regression, SelectKBest,mutual_info_classif
from scipy.stats import ttest_ind,pearsonr,spearmanr

import sklearn


def chi_squ(x_train_cat,x_test_cat,y_train):
    try:
        selector = SelectKBest(score_func=chi2, k=2)
        x_new = selector.fit_transform(x_train_cat,y_train)
        # Score card creation
        chi2_score = pd.DataFrame(
            {
                'Features': x_train_cat.columns,
                'Chi_score': selector.scores_,
                'p values': selector.pvalues_
            }
        ).sort_values(by='Chi_score', ascending=False)
        logger.info(f'----chi-square----')
        logger.info(f'Chi score data frame :\n {chi2_score}')
        chi2_score = chi2_score[chi2_score['Features'] != 'sim']
        remove_features = chi2_score[chi2_score['p values'] > 0.05]['Features']
        df_filtered = x_train_cat.drop(columns=remove_features)
        df_filtered1 = x_test_cat.drop(columns=remove_features)
        logger.info(f"Removed columns: {list(remove_features)}")
        return df_filtered,df_filtered1
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f' Issue is at line {e_linno.tb_lineno} due to {e_msg}')


def mutual(x_train_num,y_train):
    try:
        mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
        mi = mi_selector.fit(x_train_num,y_train)
        mi_scores = pd.DataFrame({
            'Feature': x_train_cat.columns,
            'MI_Score': mi_selector.scores_
        }).sort_values(by='MI_Score', ascending=False)
        remove_features = mi_scores[mi_scores['MI_Score'] > 0.05]['Feature']
        df_filtered1 = x_train_cat.drop(columns=remove_features)
        logger.info(f'---mutual----')
        logger.info(f'Removed features: {remove_features}')
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f' Issue is at line {e_linno.tb_lineno} due to {e_msg}')

def anova(x_train_num,x_test_num,y_train1):
    try:
        logger.info(f'---anaova----')
        logger.info(f"{x_train_num.info()}")

        anova_selector = SelectKBest(score_func=f_classif, k=2)
        anova_selector.fit_transform(x_train_num, y_train1)
        anova_df = pd.DataFrame({
            'Feature': x_train_num.columns,
            'F_Score': anova_selector.scores_,
            'p_value': anova_selector.pvalues_
        }).sort_values(by='F_Score', ascending=False)

        logger.info(f"ANOVA Results:\n{anova_df}")
        remove_feature = anova_df[anova_df['p_value'] > 0.05]['Feature']
        df_filter = x_train_num.drop(columns=remove_feature)
        logger.info(f'Removed features: {remove_feature}')
        #return F_Score,p_value
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f' Issue is at line {e_linno.tb_lineno} due to {e_msg}')


def ttest(p, q, r):
    try:
        ttest_results = []
        for i in p.columns:
            group1 = p[r == 0][i]
            group2 = p[r == 1][i]
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
            ttest_results.append({'Features': i, 't_stat': t_stat, 'p_value': p_val})
        ttest_df = pd.DataFrame(ttest_results).sort_values(by='t_stat', ascending=False)
        logger.info(f'----T-TEST----')
        logger.info(f"T-test results:\n{ttest_df}")
        remove_feature1 = ttest_df[ttest_df['p_value'] > 0.05]['Features']
        df_fil = p.drop(columns=remove_feature1)
        df1_fil1 = q.drop(columns=remove_feature1)
        logger.info(f"Removed columns t-test: {list(remove_feature1)}")
        return df_fil, df1_fil1
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

def co_relation(x_train_num,y_train):
    try:
        logger.info(f'---co-relation----')
        features_significant = []
        for i in x_train_num.columns:
            r, p = pearsonr(x_train_num[i], y_train)
            logger.info(f'{i}----->{r}')
            if p < 0.05:
                features_significant.append(i)
        logger.info(f'{features_significant}')


    except Exception as e:
        exc_type, exc_msg, exc_line = sys.exc_info()
        log.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')





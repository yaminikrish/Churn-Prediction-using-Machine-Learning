import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import warnings

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import minmax_scale
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import pickle
import pickle
import logging
from log_code import setup_logging
logger=setup_logging('main')
logger.info('logging successfully')
from random_sample import random_sample_imputation
from knn_imputer import knn_imp
from knn_iterater import iteration_decision_tree
from transformation import log,square_root, reci, quantile, cbrt, hyper_arsine,rank_quantile, powertrans
from visual import Visual
from outlier_handling import trim_outliers
from outlier_handling import caping
from filter_methods import chi_squ,co_relation
from algorithms import common
from parameters import para
from filter_methods import anova
from filter_methods import ttest
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from algo import common


class CHURN():
    def __init__(self,path):
        try:
            self.df=pd.read_csv(path)
            self.df=self.df.drop(['customerID','tenure','tax','gateway','JoinDate'],axis=1)
            logger.info(f'columns sucessfully droped')
            logger.info(f'Data loaded successfully:{self.df.shape}')
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            logger.info(f"Missing Value in the data : {self.df.isnull().sum()}")

            self.x=self.df.drop(['Churn'],axis=1)#independent
            logger.info(self.x.head())
            self.y=self.df['Churn']#
            logger.info(self.y.head())
            # checking if the data is clean or not:

            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,random_state=42,test_size=0.2)
            logger.info(" total:7043 --> training:5634 --> testing:1409")
        except Exception as e:
            e_type,e_msg,e_linno=sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

    '''def random(self):
        try:
            before_imp=self.x_train['TotalCharges']
            res_random = random_sample_imputation(self.x_train, self.x_test,'TotalCharges')
            if res_random is not None:
                self.x_train, self.x_test = res_random
                logger.info(f'Missing values imputed successfully')
            else:
                logger.info(f'Imputation failed')
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            before_imp.hist(bins=30, color='r')
            plt.title("Before Imputation random")
            plt.subplot(1, 2, 2)
            self.x_train['TotalCharges'].hist(bins=30, color='blue')
            plt.title("After Imputation random")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')
     def knn(self):
        before_imp_knn = self.x_train['TotalCharges']
        res_knn = knn_imp(self.x_train, self.x_test)
        if res_knn is not None:
            self.x_train, self.x_test = res_knn
            logger.info(f'Missing values imputed successfully')
        else:
            logger.info(f'Imputation failed')
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        before_imp_knn.hist(bins=30, color='r')
        plt.title("Before Imputation knn")
        plt.subplot(1, 2, 2)
        self.x_train['TotalCharges'].hist(bins=30, color='blue')
        plt.title("After Imputation knn")
        plt.tight_layout()
        plt.show()'''


    def iterator(self):
        before_imp_iter = self.x_train['TotalCharges']
        res1,res2 = iteration_decision_tree(self.x_train, self.x_test)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        before_imp_iter.hist(bins=30, color='r')
        plt.title("Before Imputation iter")
        plt.subplot(1, 2, 2)
        self.x_train['TotalCharges_itr'].hist(bins=30, color='blue')
        plt.title("After Imputation iter")
        plt.tight_layout()
        plt.show()

    def dividing(self):
        try:
            logger.info('Before dividing')
            #self.df = self.df.drop(['tenure'], axis=1)
            # Separate numerical and categorical columns
            self.x_train_num = self.x_train.select_dtypes(exclude=['object'])
            self.x_train_cat = self.x_train.select_dtypes(include=['object'])
            self.x_test_num = self.x_test.select_dtypes(exclude=['object'])
            self.x_test_cat = self.x_test.select_dtypes(include=['object'])

            logger.info(f'Numerical Column Names: {list(self.x_train_num.columns)}')
            logger.info(f'Categorical Column Names: {list(self.x_train_cat.columns)}')
            cols = ['MonthlyCharges', 'TotalCharges_itr']
            self.dummy = self.x_train[cols].copy()
            self.dummy1 = self.x_test[cols].copy()
            logger.info(f'dummy check{self.dummy.columns}')

        except Exception as e:
            import sys
            e_type, e_value, e_tb = sys.exc_info()
            logger.error(f"Issue in dividing() at line {e_tb.tb_lineno}: {e_value}")

    def variable_transform(self):
        try:
            #all1 = log(self.dummy)
            #all2=square_root(self.dummy)
            #all3=reci(self.dummy)
            #all4=cbrt(self.dummy)
            all5,all55=quantile(self.dummy,self.dummy1)
            logger.info(all5.columns)

            # if all5 is None or all55 is None:
            #     logger.info("Quantile transform failed. Returned None.")
            #     return
            # all5.index=self.x_train.index
            # all55.index=self.x_test.index
            for i in all5.columns:
                self.x_train_num[i]=all5[i]
            for i in all55.columns:
                self.x_test_num[i]=all55[i]
            #all6=hyper_arsine(self.dummy)
            #all7=rank_quantile(self.dummy)
            #all8=powertrans(self.dummy)
            #Visual.plot_transformations(all1, '_log')
            #Visual.plot_transformations(all2, '_sqrt')
            #Visual.plot_transformations(all3, '_rec')
            #Visual.plot_transformations(all4, '_cb')
            Visual.plot_transformations(all5, '_qan')
            Visual.plot_transformations(all55, '_qan')
            #Visual.plot_transformations(all6, '_hyp')
            #Visual.plot_transformations(all7, '_rank')
            #Visual.plot_transformations(all8, '_yeo')

        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')


    def handling(self):
        try:
            logger.info(f'Check - columns{self.x_train_num.columns}')
            #outliers=trim_outliers(self.x_train_num,self.x_test_num)
            a1,b1=caping(self.x_train_num,self.x_test_num,method='iqr',fold=1.5)
            #a2,b2=caping(self.x_train_num,self.x_test_num,method='gaussian',fold=2.5)
            #a3, b3 = caping(self.x_train_num, self.x_test_num, method='mad', fold=1.5)
            #a4,b4=caping(self.x_train_num,self.x_test_num, method='quantiles', fold=0.01)

            for i in a1.columns:
                self.x_train_num[i] = a1[i]
            for i in b1.columns:
                self.x_test_num[i] = b1[i]

            logger.info(f'checking that columns are saved or not {self.x_train_num.columns}')
            logger.info(f'checking that columns are saved or not {self.x_train_num.columns}')

        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f' Issue is at line {e_linno.tb_lineno} due to {e_msg}')

    def encoding_cat(self):
        try:
            #onehot encodding
            cols = [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                'PaymentMethod'
            ]

            oh = OneHotEncoder(categories='auto', drop='first', handle_unknown='ignore')
            oh.fit(self.x_train_cat[cols])
            logger.info(f'Encoder categories: {oh.categories_}')
            logger.info(f'Encoded feature names: {oh.get_feature_names_out()}')
            # Transform both datasets
            res_train = oh.transform(self.x_train_cat[cols]).toarray()
            res_test = oh.transform(self.x_test_cat[cols]).toarray()
            f_train = pd.DataFrame(res_train, columns=oh.get_feature_names_out(), index=self.x_train_cat.index)
            f_test = pd.DataFrame(res_test, columns=oh.get_feature_names_out(), index=self.x_test_cat.index)
            self.x_train_cat= pd.concat([self.x_train_cat.drop(columns=cols), f_train], axis=1)
            self.x_test_cat= pd.concat([self.x_test_cat.drop(columns=cols), f_test], axis=1)
            logger.info(f'----ONEHOT-ENCODING------')
            logger.info(f'Missing values in train:\n{self.x_train_cat.isnull().sum()}')
            logger.info(f'Missing values in test:\n{self.x_test_cat.isnull().sum()}')
            logger.info(f'Sample encoded train data:\n{self.x_train_cat.sample(5)}')
            logger.info(f'Sample encoded test data:\n{self.x_test_cat.sample(5)}')

            od = OrdinalEncoder()
            od.fit(self.x_train_cat[['Contract']])
            logger.info(f'{od.categories_}')
            logger.info(f' column_names: {od.get_feature_names_out()}')
            res_train1 = od.transform(self.x_train_cat[['Contract']])
            res_test1 = od.transform(self.x_test_cat[['Contract']])
            c_names = od.get_feature_names_out()
            f_train1 = pd.DataFrame(res_train1, columns=c_names + ['_con'], index=self.x_train_cat.index)
            f_test1 = pd.DataFrame(res_test1, columns=c_names + ['_con'], index=self.x_test_cat.index)

            self.x_train_cat = pd.concat([self.x_train_cat.drop(columns='Contract'), f_train1], axis=1)
            self.x_test_cat = pd.concat([self.x_test_cat.drop(columns='Contract'), f_test1], axis=1)
            logger.info(f'---ORDINAL-ENCODING---')
            logger.info(f'Missing values in train:\n{self.x_train_cat.isnull().sum()}')
            logger.info(f'Missing values in train:\n{self.x_test_cat.isnull().sum()}')
            logger.info(f'Sample encoded train data:\n{self.x_train_cat.sample(5)}')

    # dependent varibale should be converted using label encoder
            logger.info(f'{self.y_train[:10]}')
            lb = LabelEncoder()
            lb.fit(self.y_train)
            self.y_train = lb.transform(self.y_train)
            self.y_test = lb.transform(self.y_test)
            logger.info(f'detailed : {lb.classes_} ')
            logger.info(f'{self.y_train[:10]}')
            logger.info(f'---ORDINAL-ENCODING---')
            logger.info(f'y_train_data : {self.y_train.shape}')
            logger.info(f'y_test_data : {self.y_test.shape}')

            logger.info(f'Check null1 in before the drop {self.x_train_num["SeniorCitizen"]}')
            self.x_train_cat['SeniorCitizen'] = self.x_train_num['SeniorCitizen']
            self.x_test_cat['SeniorCitizen'] = self.x_test_num['SeniorCitizen']
            self.x_train_cat['sim']=self.x_train_cat['sim'].map({'Jio':0,'Airtel':1,'Vi':2,'BSNL':3})
            self.x_test_cat['sim'] = self.x_test_cat['sim'].map({'Jio': 0, 'Airtel': 1, 'Vi': 2, 'BSNL': 3})
            logger.info(self.x_train_cat)
            self.x_train_num = self.x_train_num.drop(['SeniorCitizen'], axis=1)
            self.x_test_num= self.x_test_num.drop(['SeniorCitizen'], axis=1)
            logger.info(f'check the null in the data frame:{self.x_train_cat.isnull().sum()}')
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f' Issue is at line {e_linno.tb_lineno} due to {e_msg}')

    def filter(self):
        try:
            x,y=chi_squ(self.x_train_cat,self.x_test_cat,self.y_train)
            self.x_train_cat1=x
            self.x_test_cat1=y
            logger.info(f'after applying chi_square filter:{self.x_train_cat1.columns}')

            cols_remove_unwanted = ['TotalCharges_itr_qan','MonthlyCharges_qan','TotalCharges_itr','TotalCharges','MonthlyCharges']
            self.x_train_num = self.x_train_num.drop(cols_remove_unwanted, axis=1)
            self.x_test_num = self.x_test_num.drop(cols_remove_unwanted, axis=1)
            logger.info(f'{self.x_train_num.isnull().sum()}')
            logger.info(self.x_train_num.isnull().sum())

            # x,y=mutual(self.x_train_cat,self.y_train)
            x=anova(self.x_train_num,self.x_test_num,self.y_train)
            x=ttest(self.x_train_num,self.x_test_num,self.y_train)
            x=co_relation(self.x_train_num,self.y_train)

        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f' Issue is at line {e_linno.tb_lineno} due to {e_msg}')

    def merge_data(self):
        try:
            # reset index so that we can concat data perfectlly
            logger.info(f'---MERGING DATA---')
            self.x_train_num.reset_index(drop=True, inplace=True)
            self.x_train_cat1.reset_index(drop=True, inplace=True)

            self.x_test_num.reset_index(drop=True, inplace=True)
            self.x_test_cat1.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.x_train_num, self.x_train_cat1], axis=1)
            self.testing_data = pd.concat([self.x_test_num, self.x_test_cat1], axis=1)

            logger.info(f'Training_data shape : {self.training_data.shape} -> {self.training_data.columns}')
            logger.info(f'Testing_data shape : {self.testing_data.shape} -> {self.testing_data.columns}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def balanced_data(self):
        try:
            logger.info('----------------Before Balancing------------------------')
            logger.info(
                f'Total row for Good category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 1)}')
            logger.info(
                f'Total row for Bad category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 0)}')
            logger.info(f'---------------After Balancing-------------------------')
            sm = SMOTE(random_state=42)
            self.training_data_res, self.y_train_res = sm.fit_resample(self.training_data, self.y_train)
            logger.info(
                f'Total row for Good category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 1)}')
            logger.info(
                f'Total row for Bad category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 0)}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


    def feature_scaling(self):
        try:
            backup_year_train = self.training_data_res['JoinYear'].copy()
            backup_year_test = self.testing_data['JoinYear'].copy()
            self.training_data_res.drop(['JoinYear'], axis=1, inplace=True)
            self.testing_data.drop(['JoinYear'], axis=1, inplace=True)
            scale_cols = ['MonthlyCharges_qan_iqr', 'TotalCharges_itr_qan_iqr']
            self.ms = StandardScaler()
            self.ms.fit(self.training_data_res[scale_cols])
            scaled_train = pd.DataFrame(
                self.ms.transform(self.training_data_res[scale_cols]),
                columns=scale_cols, index=self.training_data_res.index)
            scaled_test = pd.DataFrame(
                self.ms.transform(self.testing_data[scale_cols]),
                columns=scale_cols, index=self.testing_data.index)
            other_train = self.training_data_res.drop(scale_cols, axis=1, errors='ignore')
            other_test = self.testing_data.drop(scale_cols, axis=1, errors='ignore')
            self.training_data_res_t = pd.concat([other_train, scaled_train], axis=1)
            self.testing_data_t = pd.concat([other_test, scaled_test], axis=1)
            self.training_data_res_t['JoinYear'] = backup_year_train
            self.testing_data_t['JoinYear'] = backup_year_test
            logger.info(self.testing_data_t)
            #model = common(self.training_data_t, self.y_train_res, self.testing_data_t, self.y_test)
            logger.info("Scaling applied on MonthlyCharges_qan_iqr & TotalCharges_itr_qan_iqr.")
            with open(r"gb.pkl", "wb") as f:
                pickle.dump(self.ms, f)
            #self.training_data_t.to_csv('./Data/final.csv', index=False)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def train_models(self):
        try:
            common(self.training_data_res_t, self.y_train_res, self.testing_data_t, self.y_test)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')
    def para_selection(self):
        try:
            # self.data_ind = self.training_data_res_t.head(200)
            # self.data_dep = self.y_train_res[:200]
            # para(self.data_ind, self.data_dep)
            # logger.info(f'__________Finalized Model___________')
            # self.reg1 = LogisticRegression(C= 0.1,class_weight= None,l1_ratio=None,max_iter= 100,multi_class='auto',n_jobs=None,penalty= 'l2',solver= 'lbfgs')
            # self.reg1.fit(self.training_data_res_t, self.y_train_res)
            # logger.info(
            #     f'Train accuracy:{accuracy_score(self.y_train_res, self.reg1.predict(self.training_data_res_t))}')
            # logger.info(f'Test accuracy:{accuracy_score(self.y_test, self.reg1.predict(self.testing_data_t))}')
            # logger.info(f'=====Model Saving======')
            # with open('churn.pkl','wb') as f:
            #     pickle.dump(self.reg1,f)

            self.reg2=GradientBoostingClassifier()
            self.reg2.fit(self.training_data_res_t, self.y_train_res)
            logger.info(f'Train accuracy:{accuracy_score(self.y_train_res, self.reg2.predict(self.training_data_res_t))}')
            logger.info(f'shape:{self.training_data_res_t.shape}')
            logger.info(f'Test accuracy:{accuracy_score(self.y_test, self.reg2.predict(self.testing_data_t))}')
            logger.info(f'======MODEL SAVING======')
            with open(r'gradient_boosting.pkl','wb') as f:
                pickle.dump(self.reg2, f)
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')


if __name__=='__main__':
    try:
        path='C:\\Users\\Dell\\Downloads\\churnProject\\WA_Fn-UseC_-Telco-Customer-Churn_final.csv'
        obj=CHURN(path)
        obj.iterator()
        # #obj.knn()
        # #obj.random()
        obj.dividing()
        obj.variable_transform()
        obj.handling()
        obj.encoding_cat()
        obj.filter()
        obj.merge_data()
        obj.balanced_data()
        obj.feature_scaling()
        obj.train_models()
        obj.para_selection()


    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')
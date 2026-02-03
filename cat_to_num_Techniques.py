import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings("ignore")

from log_code import setup_logging
logger = setup_logging('cat_to_num_Techniques')

from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

class cat_to_num_Techniques:
    @staticmethod
    def label_encoding(X_train,X_test):
        try:
            logger.info('Label Encoding started')
            logger.info(f'Before Label Encoding X_train: {X_train.tail}')
            logger.info(f'Before Label Encoding X_test: {X_test.tail}')

            x_tr = X_train.copy()
            x_te = X_test.copy()

            for col in x_tr.columns:
                le = LabelEncoder()
                le.fit(pd.concat([x_tr[col], x_te[col]]).astype(str))
                x_tr[col] = le.transform(x_tr[col].astype(str))
                x_te[col] = le.transform(x_te[col].astype(str))

            logger.info('Label Encoding completed')
            logger.info(f'After Label Encoding X_train: {X_train.tail}')
            logger.info(f'After Label Encoding X_test: {X_test.tail}')

            return x_tr, x_te
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def one_hot_encoding(X_train,X_test):
        try:
            logger.info('one_hot_encoding started')
            logger.info(f'Before one_hot_encoding X_train: {X_train.tail}')
            logger.info(f'Before one_hot_encoding X_test: {X_test.tail}')

            X = pd.concat([X_train, X_test], axis=0)
            X = pd.get_dummies(X, drop_first=True)

            logger.info('one_hot_encoding completed')
            logger.info(f'After one_hot_encoding X_train: {X_train.tail}')
            logger.info(f'After one_hot_encoding X_test: {X_test.tail}')

            return X.iloc[:len(X_train),X.iloc[len(X_train):]]

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


    @staticmethod
    def frequency_encoding(X_train,X_test):
        try:
            logger.info('frequency_encoding started')
            logger.info(f'Before frequency_encoding X_train: {X_train.tail}')
            logger.info(f'Before frequency_encoding X_test: {X_test.tail}')

            x_tr = X_train.copy()
            x_te = X_test.copy()

            for col in x_tr.columns:
                freq = x_tr[col].value_counts()
                x_tr[col] = x_tr[col].map(freq)
                x_te[col] = x_te[col].map(freq).fillna(0)

            logger.info('frequency_encoding completed')
            logger.info(f'After frequency_encoding X_train: {X_train.tail}')
            logger.info(f'After frequency_encoding X_test: {X_test.tail}')

            return x_tr, x_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def binary_encoding(X_train, X_test):
        try:
            logger.debug('Binary Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            nominal_cols = X_train.select_dtypes(include='object').columns
            encoder = ce.BinaryEncoder(cols=nominal_cols)

            X_tr = encoder.fit_transform(X_train)
            X_te = encoder.transform(X_test)

            logger.debug('After Binary Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def ordinal_encoding(X_train, X_test):
        try:
            logger.debug('Ordinal Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            ordinal_cols = {
                'Contract': ['Month-to-month', 'One year', 'Two year'],
                'InternetService': ['No', 'DSL', 'Fiber optic'],
                'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
                'DeviceType': ['Old Device', 'New Device'],
                'Region': ['Rural', 'Sub Urban', 'Urban']
            }

            for col, order in ordinal_cols.items():
                if col in X_tr.columns:
                    mapping = {v: i for i, v in enumerate(order)}
                    X_tr[col] = X_tr[col].map(mapping)
                    X_te[col] = X_te[col].map(mapping)

            logger.debug('After Ordinal Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
import numpy as np
import pandas as pd
import sys
from log_code import setup_logging
logger = setup_logging("missing_values")

class MISSING_VALUE_TECHNIQUES:

    @staticmethod
    def mean_imputation(X_train, X_test):
        try:
            logger.info(f'Mean imputation for missing values')
            logger.info(f'Before imputation X_train: {X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test: {X_test.isnull().sum()}')
            mean = X_train.mean()
            logger.info(f'Mean imputation completed')
            logger.info(f'After imputation X_train: {X_train.fillna(mean).isnull().sum()}')
            logger.info(f'After imputation X_test: {X_train.fillna(mean).isnull().sum()}')
            return X_train.fillna(mean), X_test.fillna(mean)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def median_imputation(X_train, X_test):
        try:
            logger.info(f'Median imputation for missing values')
            logger.info(f'Before imputation X_train: {X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test: {X_test.isnull().sum()}')
            median = X_train.median()
            logger.info(f'Median imputation completed')
            logger.info(f'After imputation X_train: {X_train.fillna(median).isnull().sum()}')
            logger.info(f'After imputation X_test: {X_test.fillna(median).sum()}')
            return X_train.fillna(median), X_test.fillna(median)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def mode_imputation(X_train, X_test):
        try:
            logger.info(f'Mode imputation for missing values')
            logger.info(f'Before imputation X_train: {X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test: {X_test.isnull().sum()}')
            mode = X_train.mode().iloc[0]
            logger.info(f'Mode imputation completed')
            logger.info(f'After imputation X_train: {X_train.fillna(mode).isnull().sum()}')
            logger.info(f'After imputation X_test: {X_train.fillna(mode).isnull().sum()}')
            return X_train.fillna(mode), X_test.fillna(mode)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def end_of_distribution_imputation(X_train, X_test):
        eod = X_train.mean() + 3 * X_train.std()
        return X_train.fillna(eod), X_test.fillna(eod)

    @staticmethod
    def forward_fill(X_train, X_test):
        return X_train.ffill(), X_test.ffill()

    @staticmethod
    def backward_fill(X_train, X_test):
        return X_train.bfill(), X_test.bfill()

    @staticmethod
    def random_sample_imputation(X_train, X_test):
        X_tr = X_train.copy()
        X_te = X_test.copy()

        for col in X_tr.columns:
            random_values = X_tr[col].dropna()
            X_tr[col] = X_tr[col].apply(lambda x: np.random.choice(random_values) if pd.isna(x) else x)
            X_te[col] = X_te[col].apply(lambda x: np.random.choice(random_values) if pd.isna(x) else x)

        return X_tr, X_te
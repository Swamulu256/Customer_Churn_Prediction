import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging("variable_transformation_technique")
from scipy import stats
from sklearn.preprocessing import (StandardScaler,MinMaxScaler,RobustScaler,PowerTransformer,QuantileTransformer,)
from scipy.stats import boxcox

class VARIABLE_TRANSFORMATION:
    """
    This class contains different feature scaling and transformation methods.
    All transformations are applied AFTER train-test split
    to avoid data leakage.
    """

    @staticmethod
    def standard_scaling(X_train, X_test):
       try:
        logger.info('standard_scaling started')
        logger.info(f'Before X_train Data : {X_train.tail()}')
        logger.info(f'Before X_test Data : {X_test.tail()}')
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index )
        X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index )

        logger.info(f'standard_scaling Completed')
        logger.info(f'After X_train Data : {X_train_scaled.tail()}')
        logger.info(f'After X_test Data : {X_test_scaled.tail()}')

        return X_train_scaled, X_test_scaled
       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

    @staticmethod
    def minmax_scaling(X_train, X_test):
       try:
        logger.info('MinMax Scaling started')
        logger.info(f'Before X_train Data : {X_train.tail()}')
        logger.info(f'Before X_test Data : {X_test.tail()}')
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
        X_test_scaled = pd.DataFrame( scaler.transform(X_test), columns=X_test.columns,index=X_test.index )

        logger.info(f'MinMax Scaling Completed')
        logger.info(f'After X_train Data : {X_train_scaled.tail()}')
        logger.info(f'After X_test Data : {X_test_scaled.tail()}')

        return X_train_scaled, X_test_scaled
       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

    @staticmethod
    def robust_scaling(X_train, X_test):
       try:
        logger.info('robust_scaling started')
        logger.info(f'Before X_train Data : {X_train.tail()}')
        logger.info(f'Before X_test Data : {X_test.tail()}')
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index)

        logger.info(f'robust_scaling Completed')
        logger.info(f'After X_train Data : {X_train_scaled.tail()}')
        logger.info(f'After X_test Data : {X_test_scaled.tail()}')
        return X_train_scaled, X_test_scaled
       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

    @staticmethod
    def log_transform(X_train, X_test):
      try:
        logger.info('log_transform started')
        logger.info(f'Before X_train Data : {X_train.tail()}')
        logger.info(f'Before X_test Data : {X_test.tail()}')
        X_train_log = X_train.copy()
        X_test_log = X_test.copy()

        for col in X_train.columns:
            if (X_train[col] >= 0).all():
                X_train_log[col] = np.log1p(X_train[col])
                X_test_log[col] = np.log1p(X_test[col])

        logger.info('log_transform completed')
        logger.info(f'After X_train Data : {X_train_log.tail()}')
        logger.info(f'After X_test Data : {X_test_log.tail()}')

        return X_train_log, X_test_log
      except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

    @staticmethod
    def power_transform(X_train, X_test):
       try:
        logger.info('power_transform started')
        logger.info(f'Before X_train Data : {X_train.tail()}')
        logger.info(f'Before X_test Data : {X_test.tail()}')
        pt = PowerTransformer(method='yeo-johnson')
        X_train_pt = pd.DataFrame(pt.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
        X_test_pt = pd.DataFrame(pt.transform(X_test),columns=X_test.columns,index=X_test.index)

        logger.info(f'power_transform Completed')
        logger.info(f'After X_train Data : {X_train_pt.tail()}')
        logger.info(f'After X_test Data : {X_test_pt.tail()}')
        return X_train_pt, X_test_pt

       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

    @staticmethod
    def boxcox_transform(X_train, X_test):
      try:
        logger.info('boxcox_transform started')
        logger.info(f'Before X_train Data : {X_train.tail()}')
        logger.info(f'Before X_test Data : {X_test.tail()}')

        X_train_bc = X_train.copy()
        X_test_bc = X_test.copy()

        for col in X_train.columns:
            if (X_train[col] > 0).all():
                X_train_bc[col], lam = boxcox(X_train[col])
                X_test_bc[col] = boxcox(X_test[col], lmbda=lam)

        logger.info('boxcox_transform completed')
        logger.info(f'After X_train Data : {X_train_bc.tail()}')
        logger.info(f'After X_test Data : {X_test_bc.tail()}')

        return X_train_bc, X_test_bc

      except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

    @staticmethod
    def quantile_transform(X_train, X_test):
       try:
        logger.info('quantile_transform started')
        logger.info(f'Before X_train Data : {X_train.tail()}')
        logger.info(f'Before X_test Data : {X_test.tail()}')
        qt = QuantileTransformer(output_distribution='normal')

        X_train_qt = pd.DataFrame(qt.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
        X_test_qt = pd.DataFrame(qt.transform(X_test),columns=X_test.columns,index=X_test.index)
        logger.info(f'quantile_transform Completed')
        logger.info(f'After X_train Data : { X_train_qt.tail()}')
        logger.info(f'After X_test Data : {X_test_qt.tail()}')
        return X_train_qt, X_test_qt

       except Exception as e:
         error_type, error_msg, error_line = sys.exc_info()
         logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')
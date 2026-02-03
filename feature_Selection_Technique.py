import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('Feature_Selection_Techniques')
from sklearn.feature_selection import VarianceThreshold,SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class Feature_Selection_Techniques:

    @staticmethod
    def variance_threshold(x_train,x_test,threshold=0.01):
        try:
            logger.info('Variance Threshold started')
            logger.info(f'before features :{list(x_train.columns)}')
            vt = VarianceThreshold(threshold=threshold)
            vt.fit(x_train)
            cols = x_train.columns[vt.get_support()]
            logger.info(f'after features :{list(cols)}')
            return x_train[cols],x_test[cols]
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def correlation_filter(x_train, x_test, threshold=0.9):
        try:
            logger.info("Correlation Filter started")

            # Work only on numeric columns
            num_cols = x_train.select_dtypes(include='number').columns

            if len(num_cols) == 0:
                logger.warning("No numeric columns found for correlation")
                return x_train, x_test

            corr = x_train[num_cols].corr().abs()

            upper = corr.where(
                np.triu(np.ones(corr.shape), k=1).astype(bool))

            drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
            logger.info(f"Dropped columns: {drop_cols}")
            X_tr = x_train.drop(columns=drop_cols)
            X_te = x_test.drop(columns=drop_cols)

            # Preserve index alignment
            X_tr.index = x_train.index
            X_te.index = x_test.index

            return X_tr, X_te
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def select_k_best(x_train, x_test, y, k=10):
        try:
            logger.info("SelectKBest started")
            k = min(k, x_train.shape[1])
            skb = SelectKBest(score_func=chi2, k=k)
            skb.fit(abs(x_train), y)
            cols = x_train.columns[skb.get_support()]
            logger.info(f'Selected features: {list(cols)}')
            return x_train[cols], x_test[cols]

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def rfe(x_train, x_test, y):
        try:
            logger.info("RFE started")
            n_features = min(10, x_train.shape[1])
            model = LogisticRegression(max_iter=1000)
            rfe = RFE(model, n_features_to_select=n_features)
            rfe.fit(x_train, y)
            cols = x_train.columns[rfe.support_]
            logger.info(f'Selected features: {list(cols)}')
            return x_train[cols], x_test[cols]

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def lasso(x_train, x_test, y):
        try:
            logger.info("Lasso started")
            model = LogisticRegression(penalty='l1', solver='liblinear')
            model.fit(x_train, y)
            cols = x_train.columns[model.coef_[0] != 0]
            if len(cols) == 0:
                logger.info("Lasso selected no features - fallback used")
                return x_train, x_test

            logger.info(f'Selected features: {list(cols)}')
            return x_train[cols], x_test[cols]

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def tree_based(x_train, x_test, y):
        try:
            logger.info("Tree-based selection started")
            rf = RandomForestClassifier(random_state=42)
            rf.fit(x_train, y)
            imp = rf.feature_importances_
            cols = x_train.columns[imp >= np.mean(imp)]
            logger.info(f'Selected features: {list(cols)}')
            return x_train[cols], x_test[cols]

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
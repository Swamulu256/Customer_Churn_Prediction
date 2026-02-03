import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('Data_Balancing')

from imblearn.over_sampling import SMOTE, RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

class DATA_BALANCING:
    @staticmethod
    def no_balance(X, y):
        try:
            logger.info(f'Before no_balance: No.of rows for Yes class : {sum(y == 1)}')
            logger.info(f'Before no_balance: No.of rows for No class : {sum(y == 0)}')
            logger.info(f'After no_balance: No.of rows for Yes class : {sum(y == 1)}')
            logger.info(f'After no_balance: No.of rows for No class : {sum(y == 0)}')
            return X.copy(), y.copy()
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def random_over_sampling(X, y):
        try:
            logger.info(f'Before random_over_sampling: No.of rows for Yes class : {sum(y == 1)}')
            logger.info(f'Before random_over_sampling: No.of rows for No class : {sum(y == 0)}')
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X, y)
            logger.info(f'After random_over_sampling: No.of rows for Yes class : {sum(y_res == 1)}')
            logger.info(f'After random_over_sampling: No.of rows for No class : {sum(y_res == 0)}')
            return pd.DataFrame(X_res, columns=X.columns), y_res
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def random_under_sampling(X, y):
        try:
            logger.info(f'Before random_under_sampling: No.of rows for Yes class : {sum(y == 1)}')
            logger.info(f'Before random_under_sampling: No.of rows for No class : {sum(y == 0)}')
            rus = RandomUnderSampler(random_state=42)
            X_res, y_res = rus.fit_resample(X, y)
            logger.info(f'After random_under_sampling: No.of rows for Yes class : {sum(y_res == 1)}')
            logger.info(f'After random_under_sampling: No.of rows for No class : {sum(y_res == 0)}')
            return pd.DataFrame(X_res, columns=X.columns), y_res
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def smote(X, y):
        try:
            logger.info(f'Before smote: No.of rows for Yes class : {sum(y == 1)}')
            logger.info(f'Before smote: No.of rows for No class : {sum(y == 0)}')
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            logger.info(f'After smote: No.of rows for Yes class : {sum(y_res == 1)}')
            logger.info(f'After smote: No.of rows for No class : {sum(y_res == 0)}')
            return pd.DataFrame(X_res, columns=X.columns), y_res
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def smote_tomek(X, y):
        try:
            logger.info(f'Before smote_tomek: No.of rows for Yes class : {sum(y == 1)}')
            logger.info(f'Before smote_tomek: No.of rows for No class : {sum(y == 0)}')
            smt = SMOTETomek(random_state=42)
            X_res, y_res = smt.fit_resample(X, y)
            logger.info(f'After smote_tomek: No.of rows for Yes class : {sum(y_res == 1)}')
            logger.info(f'After smote_tomek: No.of rows for No class : {sum(y_res == 0)}')
            return pd.DataFrame(X_res, columns=X.columns), y_res
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def smote_enn(X, y):
        try:
            logger.info(f'Before smote_enn: No.of rows for Yes class : {sum(y == 1)}')
            logger.info(f'Before smote_enn: No.of rows for No class : {sum(y == 0)}')
            sme = SMOTEENN(random_state=42)
            X_res, y_res = sme.fit_resample(X, y)
            logger.info(f'After smote_enn: No.of rows for Yes class : {sum(y_res == 1)}')
            logger.info(f'After smote_enn: No.of rows for No class : {sum(y_res == 0)}')
            return pd.DataFrame(X_res, columns=X.columns), y_res
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
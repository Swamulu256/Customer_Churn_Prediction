import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import os
import warnings
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging("outliers_techniques")

class outlier_handling:

    # Folder to save boxplots
    plot_dir = "outlier_plots"
    os.makedirs(plot_dir, exist_ok=True)

    @staticmethod
    def winsorization(X_train, X_test):
        """
        Caps extreme values using IQR method
        """
        try:
            logger.info("Winsorization Started")
            X_tr = X_train.copy()
            X_te = X_test.copy()
            numeric_cols = X_tr.select_dtypes(include='number').columns

            for col in numeric_cols:
                Q1 = X_tr[col].quantile(0.25)
                Q3 = X_tr[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                X_tr[col] = X_tr[col].clip(lower, upper)
                X_te[col] = X_te[col].clip(lower, upper)

            logger.info("Winsorization Completed")
            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error line {error_line.tb_lineno}: {error_msg}")
            return X_train, X_test

    @staticmethod
    def robust_scaling(X_train, X_test):
        """
        Scales data using median and IQR
        """
        try:
            logger.info("Robust Scaling Started")
            scaler = RobustScaler()

            X_tr = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
            X_te = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index)
            logger.info("Robust Scaling Completed")
            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error line {error_line.tb_lineno}: {error_msg}")
            return X_train, X_test

    @staticmethod
    def log_transform(X_train, X_test):
        try:
            logger.info("Log Transform Started")
            X_tr = X_train.copy()
            X_te = X_test.copy()

            for col in X_tr.select_dtypes(include='number').columns:
                if (X_tr[col] >= 0).all():
                    X_tr[col] = np.log1p(X_tr[col])
                    X_te[col] = np.log1p(X_te[col])

            logger.info("Log Transform Completed")
            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error line {error_line.tb_lineno}: {error_msg}")
            return X_train, X_test

    def no_outlier(X_train, X_test):
        """
        Leaves data unchanged
        """
        logger.info("No Outlier Handling Applied")
        return X_train, X_test

    @staticmethod
    def save_boxplot(X_train, technique):
        try:
            for col in X_train.select_dtypes(include='number').columns:
                plt.figure(figsize=(5, 3))
                sns.boxplot(x=X_train[col])
                plt.title(f"{col} - {technique}")
                plt.tight_layout()
                path = f"{outlier_handling.plot_dir}/{technique}_{col}.png"
                plt.savefig(path)
                plt.close()

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error line {error_line.tb_lineno}: {error_msg}")
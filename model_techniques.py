import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import sys
import warnings
warnings.filterwarnings("ignore")
import pickle
from log_code import setup_logging
logger = setup_logging('Model_techniques')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

class Model_techniques:

    @staticmethod
    def knn(X_train, y_train):
       try:
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        return model
       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def naive_bayes(X_train, y_train):
       try:
        model = GaussianNB()
        model.fit(X_train, y_train)
        return model
       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def logistic(X_train, y_train):
       try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        return model
       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def decision_tree(X_train, y_train):
       try:
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model
       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def random_forest(X_train, y_train):
       try:
        model = RandomForestClassifier(n_estimators=100,random_state=42)
        model.fit(X_train, y_train)
        return model
       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def svm(X_train, y_train):
     try:
        model = SVC(probability=True)
        model.fit(X_train, y_train)
        return model

     except Exception as e:
       error_type, error_msg, error_line = sys.exc_info()
       logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def xgboost(X_train, y_train):
       try:
        model = XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    # ------------------ TUNING (ONLY USED FOR BEST MODEL) ------------------

    @staticmethod
    def tune_logistic(X_train, y_train):
       try:
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"]
        }

        grid = GridSearchCV(
            LogisticRegression(max_iter=1000),
            param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        logger.info(f"Best Logistic Params: {grid.best_params_}")
        return grid.best_estimator_
       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def tune_random_forest(X_train, y_train):
       try:
        param_grid = {"n_estimators": [100, 200],"max_depth": [None, 10, 20]}

        grid = GridSearchCV(RandomForestClassifier(random_state=42),param_grid,scoring="roc_auc",cv=3,n_jobs=-1)
        grid.fit(X_train, y_train)
        logger.info(f"Best RF Params: {grid.best_params_}")
        return grid.best_estimator_
       except Exception as e:
           error_type, error_msg, error_line = sys.exc_info()
           logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
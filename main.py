'''
In this file we are going to read data about Telco customer churn
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
import random
from scipy.stats import skew
warnings.filterwarnings('ignore')
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import pickle
from log_code import setup_logging
logger = setup_logging('main')
from visualization import (Senior_gender_vs_churn,gender_vs_churn,churn_distribution,tenure_vs_churn,monthly_charges_vs_churn,internet_service_vs_gender,contract_vs_churn,telecom_partner_vs_churn,PaymentMethod_vs_churn)
from sklearn.model_selection import train_test_split,cross_val_score
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler,Normalizer
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.model_selection import GridSearchCV
from missing_values import MISSING_VALUE_TECHNIQUES
from variable_transformation_technique import VARIABLE_TRANSFORMATION
from outliers_techniques import outlier_handling
from cat_to_num_Techniques import cat_to_num_Techniques
from feature_Selection_Technique import Feature_Selection_Techniques
from Data_Balancing import DATA_BALANCING
from model_techniques import Model_techniques
class CustomerChurn:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            random.seed(42)  # set seed fix the data constantly
            partners = ['Jio', 'Airtel', 'BSNL', 'Vodafone']  # adding telecom_partner to the data
            random_values = [random.choice(partners) for _ in range(len(self.df))]
            self.df.insert(0, 'telecom_partner', random_values)  # insert at index 0 (first column)
            logger.info(f'Data loaded Successfully')
            logger.info(f'we have : {self.df.shape[0]} Rows and {self.df.shape[1]} Columns')
            self.df = self.df.drop(['customerID'], axis=1)
            logger.info(f'\n {self.df.isnull().sum()}')
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
            logger.info(f'\n{self.df.dtypes}')
            self.X = self.df.iloc[: , :-1] #independent
            self.y = self.df.iloc[: , -1] # dependent
            self.X_train,self.X_test,self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            logger.info(f'Training Data Size : {self.X_train.shape[0]} Rows and {self.X_train.shape[1]} Columns')
            logger.info(f'Test Data Size : {self.X_test.shape[0]} Rows and {self.X_test.shape[1]} Columns')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

    def missing_value_handler(self):
       try:
        logger.info(f'Before missing values X_train: {self.X_train.isnull().sum()}')
        logger.info(f'Before missing values X_test : {self.X_test.isnull().sum()}')
        logger.info("Starting Missing Value Handling (After Split)")

        techniques = {
            'mean': MISSING_VALUE_TECHNIQUES.mean_imputation,
            'median': MISSING_VALUE_TECHNIQUES.median_imputation,
            'mode': MISSING_VALUE_TECHNIQUES.mode_imputation,
            'eod' : MISSING_VALUE_TECHNIQUES.end_of_distribution_imputation,
            'ffill': MISSING_VALUE_TECHNIQUES.forward_fill,
            'bfill': MISSING_VALUE_TECHNIQUES.backward_fill,
            'random': MISSING_VALUE_TECHNIQUES.random_sample_imputation
        }

        X_train_filled = self.X_train.copy()
        X_test_filled = self.X_test.copy()

        self.best_imputation = {}

        # Only columns with missing values
        missing_cols = self.X_train.columns[self.X_train.isnull().any()]

        for col in missing_cols:
            scores = {}

            for name, func in techniques.items():
                try:
                    X_tr_col, _ = func(
                        self.X_train[[col]].copy(),
                        self.X_test[[col]].copy())

                    # Numeric column → variance difference
                    if pd.api.types.is_numeric_dtype(self.X_train[col]):
                        original_var = self.X_train[col].var()
                        new_var = X_tr_col[col].var()
                        score = abs(original_var - new_var)
                    else:
                        # Categorical → missing reduction
                        score = abs(
                            self.X_train[col].isnull().sum() -
                            X_tr_col[col].isnull().sum())

                    scores[name] = score

                except Exception:
                    continue

            # Select best technique
            best_tech = min(scores, key=scores.get)
            self.best_imputation[col] = best_tech

            # Apply best technique
            X_train_filled[col], X_test_filled[col] = techniques[best_tech](
                self.X_train[[col]].copy(),
                self.X_test[[col]].copy())

            logger.info(f"Best imputation for {col}: {best_tech}")

        self.X_train = X_train_filled
        self.X_test = X_test_filled

        logger.info("Missing Value Handling Completed")
        logger.info(f"Train missing after:\n{self.X_train.isnull().sum()}")
        logger.info(f"Test missing after:\n{self.X_test.isnull().sum()}")

       except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(f"Error at line {error_line.tb_lineno}: {error_msg}")

    def variable_transform(self):
        try:
            logger.info("Starting Variable Transformation Selection")

            # Separate numerical & categorical features
            X_train_num = self.X_train.select_dtypes(exclude='object')
            X_test_num = self.X_test.select_dtypes(exclude='object')

            X_train_cat = self.X_train.select_dtypes(include='object')
            X_test_cat = self.X_test.select_dtypes(include='object')

            logger.info(f"Numerical Columns: {list(X_train_num.columns)}")
            logger.info(f"Categorical Columns: {list(X_train_cat.columns)}")

            # Available transformation techniques
            techniques = {
                "standard": VARIABLE_TRANSFORMATION.standard_scaling,
                "minmax": VARIABLE_TRANSFORMATION.minmax_scaling,
                "robust": VARIABLE_TRANSFORMATION.robust_scaling,
                "log": VARIABLE_TRANSFORMATION.log_transform,
                "power": VARIABLE_TRANSFORMATION.power_transform,
                "boxcox": VARIABLE_TRANSFORMATION.boxcox_transform,
                "quantile": VARIABLE_TRANSFORMATION.quantile_transform
            }

            X_train_final = X_train_num.copy()
            X_test_final = X_test_num.copy()
            best_technique_per_feature = {}

            # Loop through each numerical column
            for col in X_train_num.columns:
                logger.info(f"Evaluating transformations for column: {col}")
                skewness_scores = {}

                # Try each transformation
                for name, transform_func in techniques.items():
                    try:
                        X_tr_temp, _ = transform_func(
                            X_train_num[[col]].copy(),
                            X_test_num[[col]].copy())

                        skew_value = abs(skew(X_tr_temp[col], nan_policy='omit'))
                        skewness_scores[name] = skew_value

                    except Exception:
                        continue

                # Select best technique (minimum skew)
                best_technique = min(skewness_scores, key=skewness_scores.get)
                best_technique_per_feature[col] = best_technique

                # Apply best transformation
                X_tr_best, X_te_best = techniques[best_technique](
                    X_train_num[[col]].copy(),
                    X_test_num[[col]].copy())
                X_train_final[col] = X_tr_best[col]
                X_test_final[col] = X_te_best[col]

                logger.info(
                    f"Best transformation for {col}: {best_technique} "
                    f"(Skew: {skewness_scores[best_technique]:.4f})")

            # Combine numerical + categorical
            self.X_train = pd.concat([X_train_final, X_train_cat], axis=1)
            self.X_test = pd.concat([X_test_final, X_test_cat], axis=1)
            logger.info("Variable Transformation Completed Successfully")

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f"Error in line {error_line.tb_lineno}: {error_msg}")

    def outliers(self):
        try:
            logger.info("Starting Outlier Handling Selection")

            # Separate numerical & categorical columns
            X_train_num = self.X_train.select_dtypes(include='number')
            X_test_num = self.X_test.select_dtypes(include='number')

            X_train_cat = self.X_train.select_dtypes(include='object')
            X_test_cat = self.X_test.select_dtypes(include='object')

            # Available outlier handling techniques
            techniques = {
                'winsor': outlier_handling.winsorization,
                'robust': outlier_handling.robust_scaling,
                'log': outlier_handling.log_transform,
                'none': outlier_handling.no_outlier,
            }

            X_train_final = X_train_num.copy()
            X_test_final = X_test_num.copy()
            best_technique_per_feature = {}

            # Function to count outliers using IQR
            def count_outliers(series):
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                return ((series < lower) | (series > upper)).sum()

            # Loop through each numerical column
            for col in X_train_num.columns:
                logger.info(f"Evaluating outliers for column: {col}")

                before_count = count_outliers(X_train_num[col])
                logger.info(f"Outliers before: {before_count}")

                outlier_scores = {}

                for name, func in techniques.items():
                    try:
                        X_tr_temp, _ = func(
                            X_train_num[[col]].copy(),
                            X_test_num[[col]].copy())

                        # Count outliers after applying technique
                        outlier_count = count_outliers(X_tr_temp[col])
                        outlier_scores[name] = outlier_count

                        logger.info(f"Technique: {name} | Outliers after: {outlier_count}")

                    except Exception:
                        logger.warning(f"{name} failed for {col}: {e}")

                    if not outlier_scores:
                        logger.warning(f"No outlier technique worked for {col}. Keeping original data.")
                        continue

                # Select best technique (minimum outliers)
                best_tech = min(outlier_scores, key=outlier_scores.get)
               # best_technique_per_feature[col] = best_tech

                # Apply best technique
                X_tr_best, X_te_best = techniques[best_tech](
                    X_train_num[[col]].copy(),
                    X_test_num[[col]].copy())

                X_train_final[col] = X_tr_best[col].values
                X_test_final[col] = X_te_best[col].values

                logger.info(f"Best outlier technique for {col}: {best_tech} \n "f"(Outliers left: {outlier_scores[best_tech]})")

            # Combine numerical + categorical columns back
            self.X_train = pd.concat([X_train_final, X_train_cat], axis=1)
            self.X_test = pd.concat([X_test_final, X_test_cat], axis=1)

            logger.info("Outlier Handling Completed Successfully")

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    def cat_to_num(self):
        try:
            logger.info("Starting Encoding Cat to Num Transformation")

            x_train_cat = self.X_train.select_dtypes(include='object')
            x_test_cat = self.X_test.select_dtypes(include='object')

            x_train_num = self.X_train.select_dtypes(include='number')
            x_test_num = self.X_test.select_dtypes(include='number')

            cat_cols = x_train_cat.columns

            techniques = {
                'label' : cat_to_num_Techniques.label_encoding,
                'onehot' : cat_to_num_Techniques.one_hot_encoding,
                'frequency' : cat_to_num_Techniques.frequency_encoding,
                'binary' : cat_to_num_Techniques.binary_encoding,
                'ordinal' : cat_to_num_Techniques.ordinal_encoding
             }

            x_train_enc = pd.DataFrame(index = x_train_cat.index)
            x_test_enc = pd.DataFrame(index = x_test_cat.index)

            # Loop through each categorical column
            for col in cat_cols:
                scores = {}

                for name, func in techniques.items():
                    try:
                        X_tr_col,X_te_col = func(x_train_cat[[col]].copy(), x_test_cat[[col]].copy())
                        scores[name] = X_tr_col.shape[1]
                    except Exception:
                        continue

                 # Safety check
                if not scores:
                    logger.warning(f"No encoding worked for {col}, skipping")
                    continue

                # Select best encoding
                best = min(scores, key=scores.get)
                logger.info(f'Best encoding for {col}: {best}')

                # Apply best encoding
                X_tr_col, X_te_col = techniques[best](x_train_cat[[col]], x_test_cat[[col]])

                x_train_enc = pd.concat([x_train_enc, X_tr_col], axis=1)
                x_test_enc = pd.concat([x_test_enc, X_te_col], axis=1)

            self.X_train = pd.concat([x_train_num, x_train_enc], axis=1)
            self.X_test = pd.concat([x_test_num,  x_test_enc], axis=1)

            #self.y_train = self.y_train.loc[self.X_train.index]
            #self.y_test = self.y_test.loc[self.X_test.index]

            logger.info(f'Encoding completed. X_train shape: {self.X_train.shape} ')
        except Exception as e:
                   error_type, error_msg, error_line = sys.exc_info()
                   logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

    def feature_selection(self):
        try:
            logger.info("Starting Feature Selection")
            X_train_fs = self.X_train.copy()
            X_test_fs = self.X_test.copy()
            y = self.y_train
            techniques = {
                'variance': Feature_Selection_Techniques.variance_threshold,
                'correlation': Feature_Selection_Techniques.correlation_filter,
                'kbest': Feature_Selection_Techniques.select_k_best,
                'rfe': Feature_Selection_Techniques.rfe,
                'lasso': Feature_Selection_Techniques.lasso,
                'tree': Feature_Selection_Techniques.tree_based
            }
            scores = {}
            # Try each feature selection technique
            for name, func in techniques.items():
                try:
                    logger.info(f"Trying {name} feature selection")
                    # Handle methods with / without y
                    if name in ['variance','correlation']:
                        result = func(X_train_fs, X_test_fs)
                    else:
                        result = func(X_train_fs, X_test_fs, y)
                        # Safety check
                        if result is None:
                          logger.warning(f"{name} returned None")
                          continue
                          X_tr,X_te = result

                        # Score = number of selected features
                        scores[name] = X_tr.shape[1]
                        logger.info(f"{name} selected {X_tr.shape[1]} features")
                except Exception :
                    continue

                # Select best technique (minimum features but not zero)
                #scores = {k: v for k, v in scores.items() if v > 0}
                best_tech = min(scores, key=scores.get)
                logger.info(f"Best Feature Selection Technique: {best_tech}")

                # Apply best technique
                if best_tech == 'variance':
                    self.X_train, self.X_test = Feature_Selection_Techniques.variance_threshold(X_train_fs, X_test_fs)
                elif best_tech == 'correlation':
                    self.X_train, self.X_test = Feature_Selection_Techniques.correlation_filter(X_train_fs, X_test_fs, y)
                else:
                    self.X_train, self.X_test = techniques[best_tech](X_train_fs, X_test_fs, y)

                logger.info(f"Feature Selection Completed | "f"X_train: {self.X_train.shape}, "f"X_test: {self.X_test.shape}")

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    def data_balance(self):
        try:
            logger.info('Selecting Data Balancing Technique (ROW-WISE)')

            X = self.X_train.reset_index(drop=True)
            y = self.y_train.reset_index(drop=True)

            # HARD ASSERT
            assert len(X) == len(y), "X and y are misaligned before balancing"

            # BEFORE BALANCING
            logger.info(f'Before Balancing - Churn Distribution:\n{y.value_counts()}')

            techniques = {
                'none': DATA_BALANCING.no_balance,
                'ros': DATA_BALANCING.random_over_sampling,
                'rus': DATA_BALANCING.random_under_sampling,
                'smote': DATA_BALANCING.smote,
                'smote_tomek': DATA_BALANCING.smote_tomek,
                'smote_enn': DATA_BALANCING.smote_enn
            }
            scores = {}

            # Evaluate each balancing technique
            for name, func in techniques.items():
              try:
                logger.info(f'Applying {name} balancing')
                X_res, y_res = func(X, y)

                # Skip invalid results
                if len(X_res) != len(y_res):
                    logger.info(f'{name} skipped due to mismatch')
                    continue

                model = LogisticRegression(max_iter=1000)
                f1 = cross_val_score(model, X_res, y_res, scoring='f1', cv=3).mean()
                scores[name] = f1
                logger.info(f'{name} F1 score: {round(f1, 4)}')

              except Exception as e:
                  logger.warning(f'{name} failed: {e}')

            # Select best technique
            best = max(scores, key=scores.get)
            logger.info(f'Best Balancing Method Selected: {best}')

            # Apply best technique
            self.X_train_bal, self.y_train_bal = techniques[best](X, y)

            # AFTER BALANCING
            logger.info(f'After Balancing ({best}) - Churn Distribution:\n'f'{self.y_train_bal.value_counts()}')
            logger.info(f'Balanced X_train shape: {self.X_train_bal.shape} | 'f'Balanced y_train shape: {self.y_train_bal.shape}')

            #logger.info(f'After Balancing:\n{self.y_train_bal.value_counts()}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no :{error_line.tb_lineno}: due to {error_msg}')

    def feature_scaling(self):
        try:
            logger.info('selecting Best Feature Scaling Method')
            scalers = {
                'standard': StandardScaler(with_mean=True, with_std=True),
                'minmax': MinMaxScaler(),
                'robust': RobustScaler(),
                'maxabs': MaxAbsScaler(),
                'normalizer': Normalizer()
            }
            scores = {}
            for name, scaler in scalers.items():
                try:
                    X_scaled = scaler.fit_transform((self.X_train_bal))
                    model = LogisticRegression(max_iter=1000)
                    f1 = cross_val_score(model,X_scaled,self.y_train_bal, scoring='f1', cv=3).mean()
                    scores[name] = f1
                    logger.info(f'{name} F1 score: {round(f1, 4)}')

                except Exception as e:
                    logger.info(f'{name} failed: {e}')
            # Select best technique
            best = max(scores, key=scores.get)
            logger.info(f'Best Feature Scaling Method Selected: {best}')

            # Apply best technique
            best_scaler = scalers[best]

            self.X_train_scaled = pd.DataFrame(best_scaler.fit_transform(self.X_train_bal),columns=self.X_train_bal.columns)
            self.X_test_scaled = pd.DataFrame(best_scaler.transform(self.X_test), columns=self.X_test.columns)

            logger.info('Feature scaling applied successfully')
            with open('scaler_path.pkl', 'wb') as f:
                pickle.dump(best_scaler, f)

            logger.info(f'X_train columns : {self.X_train_scaled.columns}')
            logger.info(f'X_test columns : {self.X_test_scaled.columns}')
            # Calling common function
            #common(self.X_train_scaled, self.y_train_bal, self.X_test_scaled, self.y_test)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no :{error_line.tb_lineno}: {error_msg}')

    def Model_performance(self):
            try:
                logger.info("Starting Model Comparison")
                X_train = self.X_train_scaled
                y_train = self.y_train_bal
                X_test = self.X_test_scaled
                y_test = self.y_test
                # ------------------ STEP 1: DEFINE MODELS (PROBABILITY ENABLED) ------------------
                models = {
                    "KNN": Model_techniques.knn,
                    "NaiveBayes": Model_techniques.naive_bayes,
                    "LogisticRegression": Model_techniques.logistic,
                    "DecisionTree": Model_techniques.decision_tree,
                    "RandomForest": Model_techniques.random_forest,
                    "SVM": Model_techniques.svm,
                    "XGBoost": Model_techniques.xgboost
                }
                roc_scores = {}
                model_preds = {}
                trained_models = {}
                # ------------------ STEP 2: TRAIN & COMPUTE ROC-AUC USING PROBABILITIES ------------------
                for name, func in models.items():
                    logger.info(f"Training {name}")
                    model = func(X_train, y_train)
                    # ---- SAFETY CHECK ----
                    if model is None:
                        logger.warning(f"{name} model not returned. Skipping.")
                        continue

                    if not hasattr(model, "predict_proba"):
                        logger.warning(f"{name} does not support predict_proba(). Skipping ROC.")
                        continue

                    y_prob = model.predict_proba(X_test)[:, 1]
                    #score = roc_auc_score(y_test, y_prob)

                    roc_scores[name] = roc_auc_score(y_test, y_prob)
                    model_preds[name] = y_prob
                    trained_models[name] = model
                    logger.info(f"{name} ROC-AUC: {round(roc_scores[name], 4)}")
                    # ================== STEP 3: ROC CURVE – ALL MODELS ==================
                    plt.figure(figsize=(8, 6))
                    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

                    for name, preds in model_preds.items():
                        fpr, tpr, _ = roc_curve(y_test, preds)
                        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_scores[name]:.3f})")

                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve Comparison - All Models")
                    plt.legend(loc='lower right')
                    plt.grid(True)
                    plt.show()

                # ------------------ STEP 4: SELECT BEST MODEL ------------------
                best_model_name = max(roc_scores, key=roc_scores.get)
                best_model = trained_models[best_model_name]
                logger.info("=================================")
                logger.info(f"BEST MODEL BEFORE TUNING: {best_model_name}")
                logger.info(f"BEST ROC-AUC: {roc_scores[best_model_name]}")
                logger.info("=================================")
                # ------------------ STEP 5: TUNE ONLY BEST MODEL ------------------
                if best_model_name == "LogisticRegression":
                    logger.info("Tuning Logistic Regression")
                    best_model = Model_techniques.tune_logistic(X_train, y_train)
                elif best_model_name == "RandomForest":
                    logger.info("Tuning Random Forest")
                    best_model = Model_techniques.tune_random_forest(X_train, y_train)
                # ------------------ STEP 6: FINAL ROC CURVE (Best Model)------------------
                y_prob = best_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f"{best_model_name} (AUC={roc_auc_score(y_test, y_prob):.3f})")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC-AUC Curve")
                plt.legend()
                plt.show()
                # ------------------ STEP 6: SAVE BEST MODEL ------------------
                with open("Best_Model.pkl", "wb") as f:
                    pickle.dump(best_model, f)
                logger.info("Best model saved as Best_Model.pkl")
                return best_model

            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

if __name__ == "__main__":
    try:
        obj = CustomerChurn('C:\\Users\\DELL\\Downloads\\customer_Retention Prediction_System\\WA_Fn-UseC_-Telco-Customer-Churn (1).csv')
        gender_vs_churn(obj.df)
        churn_distribution(obj.df)
        tenure_vs_churn(obj.df)
        monthly_charges_vs_churn(obj.df)
        Senior_gender_vs_churn(obj.df)
        internet_service_vs_gender(obj.df)
        contract_vs_churn(obj.df)
        telecom_partner_vs_churn(obj.df)
        PaymentMethod_vs_churn(obj.df)

        obj.missing_value_handler()
        obj.variable_transform()
        obj. outliers()
        obj.cat_to_num()
        obj.feature_selection()
        obj.data_balance()
        obj.feature_scaling()
        obj.Model_performance()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')





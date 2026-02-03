📌 Project Overview

Customer churn is a critical problem for telecom companies. This project builds a robust, automated machine learning pipeline to analyze, preprocess, model, and predict customer churn using the Telco Customer Churn dataset.

The pipeline intelligently selects the best technique at every stage (missing values, encoding, scaling, balancing, feature selection, and modeling) based on data-driven evaluation.

🧠 Key Highlights

🔁 Fully automated ML pipeline

📊 Rich EDA & business-oriented visualizations

🧪 Technique selection instead of hard-coding

⚖️ Handles class imbalance

🏆 Compares multiple ML models

📈 Uses ROC-AUC for model selection

💾 Saves deployment-ready artifacts

🗂️ Project Structure
├── main.py
├── Visualization.py
├── missing_values.py
├── variable_transformation_technique.py
├── outliers_techniques.py
├── cat_to_num_Techniques.py
├── Feature_Selection_Techniques.py
├── Data_Balancing.py
├── Model_techniques.py
├── log_file.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_artifacts.pkl
├── scaler_path.pkl
├── final_features.pkl
└── README.md

📊 Exploratory Data Analysis (EDA)

The following visualizations are generated automatically:

Gender vs Churn

Churn Distribution

Tenure vs Churn

Monthly Charges vs Churn

Senior Citizen & Gender vs Churn

Internet Service vs Gender

Contract Type vs Churn

Telecom Partner vs Churn

Payment Method vs Churn

📌 These plots help understand customer behavior and churn drivers before modeling.

🔄 Machine Learning Pipeline
1️⃣ Data Loading & Preparation

Reads Telco churn dataset

Adds a synthetic telecom_partner feature

Converts TotalCharges to numeric

Encodes target variable (Churn: Yes → 1, No → 0)

Train–test split (80/20)

2️⃣ Missing Value Handling

Multiple imputation techniques are evaluated:

Mean

Median

Mode

End-of-Distribution

Forward Fill / Backward Fill

Random Sampling

✅ Best technique per column is selected automatically based on variance / missing reduction.

3️⃣ Variable Transformation

Numerical features are transformed using:

Standard Scaling

MinMax Scaling

Robust Scaling

Log Transform

Power Transform

Box-Cox

Quantile Transform

📉 Transformation with minimum skewness is chosen per feature.

4️⃣ Outlier Handling

Outliers are detected using IQR method and treated using:

Winsorization

Robust Scaling

Log Transform

No Treatment

🎯 The method leaving the fewest outliers is selected.

5️⃣ Categorical Encoding

Categorical variables are encoded using:

Label Encoding

One-Hot Encoding

Frequency Encoding

Binary Encoding

Ordinal Encoding

📌 Encoding is chosen based on feature dimensionality efficiency.

6️⃣ Feature Selection

Techniques evaluated:

Variance Threshold

Correlation Filter

SelectKBest

RFE

Lasso

Tree-based Selection

🏆 The technique selecting optimal minimum features is applied.
📁 Final selected features are saved as final_features.pkl.

7️⃣ Data Balancing

Class imbalance is handled using:

No balancing

Random Over Sampling

Random Under Sampling

SMOTE

SMOTE-Tomek

SMOTE-ENN

📊 Best method selected using F1-score (CV-based).

8️⃣ Feature Scaling

Scaling techniques compared:

StandardScaler

MinMaxScaler

RobustScaler

MaxAbsScaler

Normalizer

🏆 Best scaler chosen using cross-validated F1-score and saved as scaler_path.pkl.

9️⃣ Model Training & Evaluation

Models compared using ROC-AUC:

KNN

Naive Bayes

Logistic Regression

Decision Tree

Random Forest

SVM

XGBoost

📈 ROC curves are plotted for all models.

🔧 Hyperparameter Tuning

Only the best performing model is tuned:

Logistic Regression → GridSearchCV

Random Forest → GridSearchCV

🏆 Final Output

✅ Best trained model

✅ Final feature list

✅ Scaler

✅ ROC-AUC performance


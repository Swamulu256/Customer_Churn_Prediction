import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import logging
import warnings
import sys
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging('Visualization')

def gender_vs_churn(df):
    try:
        logger.info("Plotting gender vs churn")
        ax = pd.crosstab(df['gender'], df['Churn']).plot(kind='bar')
        plt.xlabel("Gender")
        plt.ylabel("Number of Customers")
        plt.title("Gender vs Churn")
        for container in ax.containers:
            ax.bar_label(container)
        plt.legend(title="Churn")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

def churn_distribution(df):
    try:
        logger.info("Plotting churn distribution")
        churn = df['Churn'].value_counts()
        fig, ax = plt.subplots()
        bars = ax.bar(churn.index, churn.values)
        ax.set_xlabel("Churn")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customer Churn Distribution")
        ax.bar_label(bars)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

def tenure_vs_churn(df):
    try:
        logger.info("Plotting tenure vs churn")
        churn_yes = df[df['Churn'] == 'Yes']['tenure']
        churn_no = df[df['Churn'] == 'No']['tenure']
        plt.hist(churn_yes, bins=30, alpha=0.6)
        plt.hist(churn_no, bins=30, alpha=0.6)
        plt.xlabel("Tenure (Months)")
        plt.ylabel("Customers")
        plt.title("Tenure vs Churn")
        plt.legend(["Churn Yes", "Churn No"])
        plt.show()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

def monthly_charges_vs_churn(df):
    try:
        logger.info("Plotting monthly charges vs churn")
        churn_yes = df[df['Churn'] == 'Yes']['MonthlyCharges']
        churn_no = df[df['Churn'] == 'No']['MonthlyCharges']
        plt.hist(churn_yes, bins=30, alpha=0.6)
        plt.hist(churn_no, bins=30, alpha=0.6)
        plt.xlabel("Monthly Charges")
        plt.ylabel("Customers")
        plt.title("Monthly Charges vs Churn")
        plt.legend(["Churn Yes", "Churn No"])
        plt.show()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

def Senior_gender_vs_churn(df):
    try:
        logger.info("Plotting SeniorCitizen vs churn")
        senior_gender = df[df['SeniorCitizen'] == 1]
        logger.info(f"Senior_Gender vs Churn :\n{senior_gender}")
        ax = pd.crosstab(senior_gender['gender'], senior_gender['Churn']).plot(kind='bar')
        plt.title("Churn among Senior Citizens by Gender")
        for container in ax.containers:
            ax.bar_label(container)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

def PaymentMethod_vs_churn(df):
    try:
     plt.figure(figsize=(6,4))
     sns.countplot(data=df, x='PaymentMethod', hue='Churn')
     plt.title('Payment Method vs Churn')
     plt.xticks(rotation=45)
     plt.tight_layout()
     plt.show()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

def internet_service_vs_gender(df):
   try:
    logger.info("Plotting Internet Service vs Gender")
    service_gender = pd.crosstab(df['InternetService'], df['gender'])
    logger.info(f"Internet Service vs Gender Crosstab:\n{service_gender}")
    ax = service_gender.plot(kind='bar')
    plt.xlabel("Internet Service")
    plt.ylabel("Number of Customers")
    plt.title("Internet Service vs Gender")
    plt.legend(title="Gender")
    for container in ax.containers:
        ax.bar_label(container, label_type='edge')
    plt.tight_layout()
    plt.show()
   except Exception as e:
       error_type, error_msg, error_line = sys.exc_info()
       logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

def contract_vs_churn(df):
  try:
    logger.info("Plotting Contract vs Churn")
    contract_churn = pd.crosstab(df['Contract'], df['Churn'])
    logger.info(f"Contract vs Churn Crosstab:\n{contract_churn}")
    ax = contract_churn.plot(kind='bar')
    plt.xlabel("Contract Type")
    plt.ylabel("Number of Customers")
    plt.title("Contract vs Churn")
    plt.legend(title="Churn")
   # Add values on bars
    for container in ax.containers:
        ax.bar_label(container, label_type='edge')
    plt.tight_layout()
    plt.show()
  except Exception as e:
      error_type, error_msg, error_line = sys.exc_info()
      logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

def telecom_partner_vs_churn(df):
  try:
        logger.info("Plotting TelecomPartner vs Churn")
        TelecomPartner_churn = pd.crosstab(df['telecom_partner'], df['Churn'])
        logger.info(f"TelecomPartner vs Churn Crosstab:\n{TelecomPartner_churn}")
        ax = TelecomPartner_churn.plot(kind='bar')
        plt.xlabel("TelecomPartner Type")
        plt.ylabel("Number of Customers")
        plt.title("TelecomPartner vs Churn")
        for container in ax.containers:
            ax.bar_label(container, label_type='edge')
        plt.legend(title="Churn")
        plt.tight_layout()
        plt.show()
  except Exception as e:
      error_type, error_msg, error_line = sys.exc_info()
      logger.info(f'error in line no :{error_line.tb_lineno}: due to {error_msg}')

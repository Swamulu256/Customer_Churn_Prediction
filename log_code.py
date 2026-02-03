import logging
import logging

def setup_logging(script_name):
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # 🚨 Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create a file handler for the script
    handler = logging.FileHandler(f'C:\\Users\\DELL\\Downloads\\customer_Retention Prediction_System\\logs\\{script_name}.log',mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
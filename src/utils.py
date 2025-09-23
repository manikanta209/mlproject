import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
import dill


# Function to save an object to a file using dill
def save_object(file_path, obj):
    try:

        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and dump the object
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

# Function to evaluate multiple models and return their R2 scores
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            # Fit the model
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Calculate R2 score, r2_score means "coefficient of determination"
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
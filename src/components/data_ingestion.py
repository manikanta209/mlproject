import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.exception import CustomException

from src.components.data_tranformation import DataTransformation
from src.components.data_tranformation import DataTransformationConfig
from src.utils import save_object

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

""" 
  1.Configuration for data ingestion paths
  2.This class holds the file paths for training, testing, and raw data.
  3.The paths are set to be within an 'artifacts' directory.
  4.The dataclass decorator automatically generates special methods like __init__() and __repr__(). 
"""
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')



""" This method initiates the data ingestion process. I
    It reads a CSV file, splits the data into training and testing sets, and saves these sets to
    the paths defined in the DataIngestionConfig class. 
    It also includes logging for tracking the process and error handling to raise a custom exception in case of failure. """

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv') # Read the dataset
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # Create directory if not exists
 
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # Save raw data
            logging.info("Raw data is saved")

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) # Split the data into train and test sets

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) # Save train data 
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) # Save test data

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception occurred at data ingestion stage")
            raise CustomException(e, sys)  
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ =  data_transformation.initiate_data_transformation(train_data, test_data)

    modelTrainer = ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_arr, test_arr))



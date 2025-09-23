import sys
from dataclasses import dataclass
import os 

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

""" 
@dataclass: This decorator automatically generates special methods like __init__(), __repr__(), etc., for the DataTransformationConfig class. 
This saves you from writing boilerplate code 
"""

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl') # Path to save the preprocessor object


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() # Initialize the configuration

    """
    This function reads training and testing data from given file paths, applies data transformation using a preprocessor object,
    and saves the transformed data and preprocessor object to specified file paths.
    """
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            logging.info("Data Transformation initiated")

            # Define which columns should be ordinal-encoded and which should be scaled

            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [ 
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]   

            # Numerical pipeline
            # This pipeline handles missing values and scales numerical features

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # Fill missing values with median    
                 ('scaler', StandardScaler()) # Scale features to have mean=0 and variance=1 , values will usually fall between -3 and +3 
            ])     

            # Categorical pipeline
            # This pipeline handles missing values, encodes categorical features, and scales them

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # Fill missing values with the most frequent value
                ('one_hot_encoder', OneHotEncoder()), # Converts categorical variables (text/labels) into a binary (0/1) matrix. 
                ('scaler', StandardScaler(with_mean=False)) # Scale features to have mean=0 and variance=1 , values will usually fall between -3 and +3 
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Combine both numerical and categorical pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer(   
                [
                    ('num_pipeline', num_pipeline, numerical_columns), 
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
            logging.info("Data Transformation completed")



        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_object() # Get the preprocessor object

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            # Separate input features (X) and target (y) for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features (X) and target (y) for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply the preprocessor object to the input features of training and testing data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            # Combine the transformed input features with the target feature for both training and testing data

            train_arr =  pd.concat(
                [pd.DataFrame(input_feature_train_arr), #  Transformed input features (X_train)
                 pd.DataFrame(target_feature_train_df.reset_index(drop=True))],  # Target column (y_train)
                 axis=1)
            
            test_arr =  pd.concat(
                [pd.DataFrame(input_feature_test_arr), #  Transformed input features (X_test)
                 pd.DataFrame(target_feature_test_df.reset_index(drop=True))],  # Target column (y_test)
                 axis=1)

            # Save the preprocessor object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
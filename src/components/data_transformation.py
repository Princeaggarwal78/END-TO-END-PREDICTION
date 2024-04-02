from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import sys
import numpy as np
from src.utils import save_object

@dataclass
class Data_Transformation_config:

    preprocessor_obj_file_path = os.path.join("artifact","preprocessor.pkl")

class Data_Transformation:

    def __init__(self):
        
        self.data_transformation = Data_Transformation_config()
       
    def get_data_transformation(self):
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ('standardscaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ('impute',SimpleImputer(strategy="most_frequent")),
                    ("encoding",OneHotEncoder()),
                    ('standardscaler',StandardScaler(with_mean=False))
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("data reading started")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            input_features = "math score"

            train_transformation = train_df.drop(columns=[input_features],axis = 1)
            train_label = train_df[input_features]

            test_transformation = test_df.drop(columns=[input_features],axis = 1)
            test_label = test_df[input_features]

            preprocessing_obj = self.get_data_transformation()
            logging.info("data preprocessing is started")
            train_arr_pre = preprocessing_obj.fit_transform(train_transformation)
            test_arr_pre = preprocessing_obj.transform(test_transformation)

            train_arr = np.c_[
                train_arr_pre,np.array(train_label)
            ]
            test_arr = np.c_[
                test_arr_pre,np.array(test_label)
            ]
            logging.info("saving and return file in data_transformation")
            save_object(
                file_path = self.data_transformation.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

        
                 
    
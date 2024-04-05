import os 
import sys 
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.utils import model_evaluation
from src.utils import save_object
from sklearn.metrics import r2_score


@dataclass
class Model_Trainer_Config:

    trainer_model = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.trainer_config = Model_Trainer_Config()

    def model_training(self,train_arr,test_arr):
            try:
                logging.info("Entered in model training")
                logging.info("TRAIN TEST SPLIT")
                x_train,x_test,y_train,y_test = (
                     train_arr[:,:-1],test_arr[:,:-1],train_arr[:,-1],test_arr[:,-1]
                )

                models = {                
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                }
                logging.info("best model searching")
                model_report:dict = model_evaluation(x_train = x_train,y_train = y_train,x_test = x_test,y_test = y_test,models = models)
                logging.info("")
                best_model_value = max(sorted(model_report.values()))
                best_model_name = list(model_report.keys())[
                     list(model_report.values()).index(best_model_value)
                     ]
                best_model = models[best_model_name]
                if best_model_value<0.6 :
                     raise CustomException("no best model found")
                                   
                save_object(
                     file_path= self.trainer_config.trainer_model,
                     obj= best_model
                )
                logging.info("prediction started")
                predicted = best_model.predict(x_test)
                r2_square = r2_score(y_test,predicted)
                
                return r2_square

            except Exception as e:
                 raise CustomException(e,sys)

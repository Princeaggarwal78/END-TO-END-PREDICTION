import os 
import sys
from src.exception import CustomException
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try: 
        direc_name = os.path.dirname(file_path)
        os.makedirs(direc_name,exist_ok=True)

        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
 
def model_evaluation(x_train,y_train,x_test,y_test,models,params):

    try:
        
        report = {}
        for i in range(len(list(models))):

            param = params[list(models.keys())[i]]
            model = list(models.values())[i]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_test_predict = model.predict(x_test)
            report[list(models.keys())[i]] = r2_score(y_test,y_test_predict)
            
        return report

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)


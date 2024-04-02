import os 
import sys
from src.exception import CustomException
import pandas as pd
import numpy as np
import dill

def save_object(file_path,obj):
    try: 
        direc_name = os.path.dirname(file_path)
        os.makedirs(direc_name,exist_ok=True)

        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
 
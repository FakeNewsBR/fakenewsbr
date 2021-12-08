import pandas as pd
import sqlite3
import numpy as np
import pickle
import joblib

def getrandomfake():
    with sqlite3.connect('fakenewsbr.db') as con:
        fakes = pd.read_sql("Select desc_full_text from tblFakeNews",con)
        return fakes.iloc[np.random.randint(fakes.shape[0]),0] 

def getfakequantity():
    with sqlite3.connect('fakenewsbr.db') as con:
        fakes = pd.read_sql("Select desc_full_text from tblFakeNews",con)
        return fakes.shape[0]

def getfakedatabase():
    with sqlite3.connect('fakenewsbr.db') as con:
        fakes = pd.read_sql("Select * from tblFakeNews",con)
    return fakes.to_json(orient='records')

def getmodelsdataframe():
    with sqlite3.connect('fakenewsbr.db') as con:
        df_models = pd.read_sql("Select * from tblModels",con)
        return df_models

def getmodelslistname(df, nm_model):
    return df[nm_model].values

def getmodelslistid(df, id_model):
    return df[id_model].values

def evaluatemodel(text, file_model):
    #source https://realpython.com/python-import/
    #import importlib
    #Carrega o modelo    
    #modelo = importlib.import_module('modelos/'+file_model)
    print(f"**************Nome do Arquivo: {file_model} *********{__name__}****************")
    model = joblib.load("modelos/"+file_model)
    return model.predict(text)





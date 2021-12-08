import joblib


def evaluatemodel(text, file_model):
    #source https://realpython.com/python-import/
    #import importlib
    #Carrega o modelo    
    #modelo = importlib.import_module('modelos/'+file_model)
    print(f"**************Nome do Arquivo: {file_model} *********{__name__}****************")
    model = joblib.load("modelos/"+file_model)
    return model.predict(text)
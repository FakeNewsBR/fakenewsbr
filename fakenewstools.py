import pandas as pd
import sqlite3
import numpy as np


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



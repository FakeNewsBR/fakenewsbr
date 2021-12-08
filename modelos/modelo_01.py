import pandas as pd
import pickle
from nltk.stem import SnowballStemmer
import lightgbm as lgbm
import nltk as nltk

def modelo(novo_texto):
    """
    Replicando os tratamentos que foram feitos para desenvolver o modelo, carregando o modelo que foi treinado
    e escorando.
    """
    stemmer=SnowballStemmer('portuguese')
    stopwords = nltk.corpus.stopwords.words('portuguese')
    
    output=pd.DataFrame({'texto':[novo_texto]})
    data=output.copy()
    
    data['limpa']=data['texto'].apply(lambda x:" ".join([stemmer.stem(i) for i in x.split() if i not in stopwords]).lower())
    data['Uppercase'] = data['texto'].str.findall(r'[A-Z]').str.len()
    data['Lowercase'] = data['texto'].str.findall(r'[a-z]').str.len()
    data=data.drop(columns=['texto'])
    data['Total'] = data['limpa'].str.findall(r'[a-z]').str.len()
    data['limpa']=["".join(limpa) for limpa in data['limpa'].values]
    
    data['Razao_Upper_Lower']=data['Uppercase']/data['Lowercase']
    data['Razao_Upper_Total']=data['Uppercase']/data['Total']
    
    data['limpa']=data['limpa'].str.replace("ão","ao")
    data['limpa']=data['limpa'].str.replace("á","a")
    data['limpa']=data['limpa'].str.replace("é","e")
    data['limpa']=data['limpa'].str.replace("í","i")
    data['limpa']=data['limpa'].str.replace("ó","o")
    data['limpa']=data['limpa'].str.replace("ô","o")
    data['limpa']=data['limpa'].str.replace("ú","u")
    data['limpa']=data['limpa'].str.replace("ão","ao")
    data['limpa']=data['limpa'].str.replace("ó","o")
    data['limpa']=data['limpa'].str.replace("é","e")
    data['limpa']=data['limpa'].str.replace(".","")
    data['limpa']=data['limpa'].str.replace(";","")
    data['palavra_7exclamacoes']=data.limpa.str.count(r'!!!!!!!')
    data['palavra_3exclamacoes']=data.limpa.str.count(r'!!!')
    data['palavra_...']=data.limpa.str.count(r'...')
    data['palavra_.....']=data.limpa.str.count(r'.....')
    
    data = data[['Uppercase', 'Lowercase', 'limpa', 'Total', 'Razao_Upper_Lower',
       'Razao_Upper_Total', 'palavra_7exclamacoes', 'palavra_3exclamacoes',
       'palavra_...', 'palavra_.....']]

    #Carregando modelo
    with open('modelos/modelo_lightgbm14102021.pkl', 'rb') as f:
        modelo_carregado=pickle.load(f)

    result = modelo_carregado.predict_proba(data)[0][0]

    return(result)
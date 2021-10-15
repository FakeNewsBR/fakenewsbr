import pandas as pd
import pickle
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgbm
import nltk as nltk

def modelo(novo_texto):
    """
    Replicando os tratamentos que foram feitos para desenvolver o modelo, carregando o modelo que foi treinado
    e escorando.
    """
    output=pd.DataFrame({'texto':[novo_texto]})
    data=output.copy()
    stemmer=SnowballStemmer('portuguese')
    stopwords = nltk.corpus.stopwords.words('portuguese')
    
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
    
    transformador1 = TfidfVectorizer(ngram_range=(1,2),stop_words=stopwords,sublinear_tf=True,max_features=2000,strip_accents='ascii')
    passo1_a = transformador1.fit_transform(data['limpa'])
    Passo1=pd.DataFrame(passo1_a.toarray(),columns=transformador1.get_feature_names())
    
    transformador2 = CountVectorizer(ngram_range=(1,2),stop_words=stopwords,max_features=2000,strip_accents='ascii')
    passo2_a = transformador2.fit_transform(data['limpa'])
    Passo2=pd.DataFrame(passo2_a.toarray(),columns=transformador2.get_feature_names())
    
    transformador3 = CountVectorizer(ngram_range=(1,3),stop_words=stopwords,max_features=500,strip_accents='ascii')
    passo3_a = transformador3.fit_transform(data['limpa'])
    transformador3.get_feature_names() 
    Passo3=pd.DataFrame(passo3_a.toarray(),columns=transformador3.get_feature_names())
    
    frames = [data,Passo1,Passo2,Passo3]
    Base_Final = pd.concat(frames,axis=1)
    
    #Carregando modelo
    modelo_carregado=pickle.load(open("modelos/Modelo_lightgbm061021.sav", 'rb'))
    
    ordem_variaveis=['Uppercase',
        'Lowercase',
        'Total',
        'Razao_Upper_Lower',
        'Razao_Upper_Total',
        'palavra_...',
        'palavra_.....',
        'acus',
        'afirm',
        'agor',
        'ai',
        'alto',
        'ano',
        'apos',
        'atraves',
        'autor',
        'bairr',
        'bem',
        'brasil',
        'brasileiros',
        'cam',
        'canc',
        'car',
        'cheg',
        'cont',
        'defes',
        'desd',
        'dess',
        'dest',
        'dia',
        'dilm',
        'diss',
        'distrit',
        'durant',
        'entanto',
        'escond',
        'estud',
        'evident',
        'ex',
        'ex govern',
        'exist',
        'fal',
        'famil',
        'feir',
        'feira',
        'form',
        'g1',
        'ganh',
        'gent',
        'geral',
        'ha',
        'hoj',
        'hoje',
        'homens',
        'impost',
        'intern',
        'ja',
        'jornal',
        'lav',
        'lav jato',
        'lei',
        'lug',
        'lul',
        'maior',
        'manha dest',
        'mar',
        'marc',
        'mes',
        'milho',
        'milit',
        'ministr',
        'nest',
        'noss',
        'outr',
        'pais',
        'par',
        'part',
        'paul',
        'ped',
        'pel',
        'pen',
        'pesquis',
        'pod',
        'poder',
        'polic',
        'polit',
        'politico',
        'possu',
        'preso',
        'process',
        'psdb',
        'public',
        'quarta',
        'quas',
        'quinta',
        'receb',
        'regia',
        'regional',
        'represent',
        'respons pel',
        'rio',
        'sa',
        'segund',
        'send',
        'senhor',
        'ser',
        'servic',
        'sit',
        'so',
        'sob',
        'sobr',
        'suprem',
        'tal',
        'tambem',
        'tent',
        'ter',
        'terror',
        'testemunh',
        'tip',
        'tod',
        'torn',
        'trat',
        'tribunal',
        'tribunal federal',
        'tud',
        'ultim',
        'vez',
        'voc']
    #Caso não haja a palavra no texto temos que criar a coluna:
    for variavel in ordem_variaveis:
        if variavel not in Base_Final.columns:
            Base_Final[variavel]=0

    Base_Final=Base_Final.loc[:,~Base_Final.columns.duplicated()]    
    return(modelo_carregado.predict_proba(Base_Final[ordem_variaveis])[:,0])


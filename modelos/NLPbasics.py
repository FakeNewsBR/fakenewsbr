import pandas as pd
import numpy as np
import unidecode
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from itertools import chain #Para a função oov
from scipy.special import softmax 

class NLPbasics():
    def __init__(self):
        """
        Inicialização da classe NeoNLP
        Atributo:
        text2vec = CountVectorizer
        """
        self.text2vec = CountVectorizer(strip_accents='unicode')
        self.text2tfidf = TfidfTransformer()
        self.default_method = 'tf'

        
    def normalize(self, text):
        """
        função normalize
        Transforma texto para minúsculo e sem acento
        Entrada:
        texto
        Retorno:
        Texto no formato minúsculo e sem acento.
        """
        return unidecode.unidecode(text.lower())
    
    def tokenize(self, text):
        """
        Função tokenize
        Divide o texto em palavras utilizando como separador espaço em branco (múltiplos espaços são considerados como um)
        """
        return text.split(' ')
    
    def normalize_tokenize(self, text):
        """
        Função normalize_tokenize
        Aplica na sequencia a função normalize e depois a função tokenize
        """
        return self.tokenize(self.normalize(text))

    #def fit(self, corpus, categories):
    #    self.df_model = pd.DataFrame(data = [corpus, categories], columns = ['corpus','categories'])
    
    def build_model(self,corpus):
        """
        Função build_model
        Cria o vocabulário a partir de um corpus gerando o dicionário de-para
        Armazena o corpus utilizado no atributo corpus
        """
        self.corpus = corpus.copy()
        self.text2vec.fit(self.corpus)

    def fit(self, description, classification):
        #Prepara o dataframe
        self.df_classification = pd.DataFrame({'description':description,'category':classification})
        self.df_group = self.df_classification.groupby("category").agg({'description':lambda x: ' '.join(x)})
        self.A = self.transform(self.df_group.description.values).T
        self.B = self.text2tfidf.fit_transform(self.A.T).T.toarray()        
        #idf = self.A.count_nonzero
        self.cat2index = dict(zip(self.df_group.index, range(len(self.df_group))))
        self.index2cat = dict(zip(range(len(self.df_group)),self.df_group.index))

    def predict(self, text):
        """
        Função predict
        Realiza a predição de uma categoria a partir de uma base de treinamento
        Requisitos:
            Requer a criação do vocabulário (função build_model)
            Requer antes a chamada da função fit, passando as descrições e as classes (função fit)
        Argumentos:
            category: True or False. por padrão True, retorna o nome da categoria, caso False retorna o número da categoria
            para recuperar o nome pode-se chamar a função index2cat para recuperar o nome da categoria
            method: 
                'tf': utiliza diretamente a matriz termo frequencia para recuperar a categoria, realizando o produto A.T@x e encontrando o argmax
                'tf_norm': utiliza o vetor tf normalizado (A/np.linalg.norm(A,axis=0)).T@x e encontrando o argmax
                'cossim': utiliza a similaridade a partir do cálculo do cosseno (A/np.linalg.norm(A,axis=0)).T@(x/np.linalg.norm(x)) e devolve o argmax
                'tfidf': utiliza o modelo tf normalizado com idf suavizado
                'prob':utiliza um modelo probabilístico orientado por linha

        """
        method = self.default_method
        x = self.transform(text).T
        if method == 'tf':
            result = self.A.T@x
        elif method == 'tf_norm':
            result = (self.A/np.linalg.norm(self.A,axis=0)).T@x
        elif method == 'cossim':
            result = (self.A/np.linalg.norm(self.A,axis=0)).T@(x/np.linalg.norm(x))
        elif method == 'tfidf':
            result = self.B.T@x
        elif method == 'prob':
            result = np.exp(np.log((self.A/self.A.sum(axis=1).reshape(-1,1)) + 1/self.A.sum()).T@x)
        else:
            raise f"method {method} invalid, try ['tf','tf_norm','cossim']" 
        
        return (result/result.sum())[0][0]
        #pd.DataFrame({'category':neonlp.cat2index.keys(),'score':(neonlp.A.T@x).T[0]}).sort_values('score',ascending=False)

    
    def oov(self, raw_documents):
        analyzer = self.text2vec.build_analyzer()
        analyzed_documents = [analyzer(doc) for doc in (raw_documents if type(raw_documents)!=type('') else [raw_documents]) ]
        new_tokens = set(chain.from_iterable(analyzed_documents))
        oov_tokens = new_tokens.difference(set(self.text2vec.vocabulary_.keys()))
        return oov_tokens

    def predict2df(self, text, category = True, method = 'tf'):
        """
        Função predict
        Realiza a predição de uma categoria a partir de uma base de treinamento
        Requisitos:
            Requer a criação do vocabulário (função build_model)
            Requer antes a chamada da função fit, passando as descrições e as classes (função fit)
        Argumentos:
            category: True or False. por padrão True, retorna o nome da categoria, caso False retorna o número da categoria
            para recuperar o nome pode-se chamar a função index2cat para recuperar o nome da categoria
            method: 
                'tf': utiliza diretamente a matriz termo frequencia para recuperar a categoria, realizando o produto A.T@x e encontrando o argmax
                'tf_norm': utiliza o vetor tf normalizado (A/np.linalg.norm(A,axis=0)).T@x e encontrando o argmax
                'cossim': utiliza a similaridade a partir do cálculo do cosseno (A/np.linalg.norm(A,axis=0)).T@(x/np.linalg.norm(x)) e devolve o argmax

        """
        x = self.transform(text).T
        if method == 'tf':
            result = self.A.T@x
        elif method == 'tf_norm':
            result = (self.A/np.linalg.norm(self.A,axis=0)).T@x
        elif method == 'cossim':
            result = (self.A/np.linalg.norm(self.A,axis=0)).T@(x/np.linalg.norm(x))
        elif method == 'tfidf':
            result = self.B.T@x
        else:
            raise f"method {method} invalid, try ['tf','tf_norm','cossim','tfidf']"         
        return pd.DataFrame({'category':self.cat2index.keys(),'score':result.T[0]}).sort_values('score',ascending=False)
        
    def transform(self, text, toarray=True):
        """
        Função transform
        Retorna a representação bag of words de um texto ou um vetor de um texto
        Realiza por padrão a normalização (para minúsculo e retira acentos) e a tokenização por espaço
        Argumentos:
        text = texto a transformar
        toarray = por padrão é o valor de retorno, se False retorno no formato de matriz esparsa
        """
        if type(text)==type(''):        
            bagofwords = self.text2vec.transform([text])
        else: 
            bagofwords = self.text2vec.transform(text)
        return bagofwords.toarray() if toarray else bagofwords 
            
        
    def cos_similarity(self, text1, text2):
        vector01 = self.transform(text1)
        vector02 = self.transform(text2)
        return vector01@vector02/(np.linalg.norm(vector01) * np.linalg.norm(vector02))

    def vocabulary(self):
        """
        Função vocabulary
        retorna um dicionário com o vocabulário gerado e seu índice
        """
        return self.text2vec.vocabulary_

    def token2index(self, token):
        return self.text2vec.vocabulary_.get(token)

    def index2token(self, index):
        return dict(zip(self.text2vec.vocabulary_.values(), self.text2vec.vocabulary_.keys())).get(index)

    #def most_similar(self, query):        
    

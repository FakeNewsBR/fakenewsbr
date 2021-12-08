
from flask import Flask, render_template, request
import fakenewstools as fn
import modelos.modelo_01 as modelo
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import joblib

#app = Flask(__name__,template_folder="/home/fakenewsbr/mysite/templates")
#app = Flask(__name__)#, template_folder = r"D:\Nuvem\ghdaru\OneDrive\030_DOUTORADO\460_MECAI\020_Probabilidade e Estatística\010_PROJETOFAKENEWS\fakenewsbr\templates")
app = Flask(__name__, template_folder = r"D:\Nuvem\ghdaru\OneDrive\030_DOUTORADO\460_MECAI\020_Probabilidade e Estatística\010_PROJETOFAKENEWS\fakenewsbr\templates")
print("Inicializando ****************",__name__)

content_index = {
    'text':'',
    'news':'',
    'database_size':fn.getfakequantity(),
    'vl_true':50,
    'vl_false':50,
    'results': '/static/img/barplot.png',
    'models_df': fn.getmodelsdataframe(),
    'models':fn.getmodelslistname(fn.getmodelsdataframe(),'nm_model'),
}

@app.route('/index.html')
@app.route('/data')
@app.route('/')
def main():    
    return render_template('index.html', **content_index)

@app.route('/news')
def main_news():
    content_index['news'] = fn.getrandomfake()
    content_index['vl_false'] = 50
    content_index['vl_true'] = 50
    return render_template('index.html', **content_index)

@app.route('/verificar', methods=['POST'])
def verificar():
    if request.method == 'POST':
        current_text = " optClassifier: " +  request.form.get('optClassifier','0')
        content_index['news'] = request.form.get('news',content_index['news'])        
        content_index['vl_false'] = int(modelo.modelo(content_index['news'])*100)
        content_index['vl_true'] = 100 - content_index['vl_false']

        resultados = {"modelo":[],'fake':[],'true':[]}

        for model in content_index['models_df'].nm_file.values:
            try:
                print(f'Model:{model}')
                prob = fn.evaluatemodel(content_index['news'], model)
                print(f'Model:{model}, probabilidade:{prob}')
            except Exception as error:
                print(f'Model:{model} não executado {error} ')

        fig, ax = plt.subplots(figsize=(5,5))
        sns.set_color_codes("pastel")
        tips = sns.load_dataset("tips")
        total = tips.groupby('day')['total_bill'].sum().reset_index()
        smoker = tips[tips.smoker=='Yes'].groupby('day')['total_bill'].sum().reset_index()
        smoker['total_bill'] = [i / j * 100 for i,j in zip(smoker['total_bill'], total['total_bill'])]
        total['total_bill'] = [i / j * 100 for i,j in zip(total['total_bill'], total['total_bill'])]
        # bar chart 1 -> top bars (group of 'smoker=No')
        bar1 = sns.barplot(y="day",  x="total_bill", data=total, color='b')

        # bar chart 2 -> bottom bars (group of 'smoker=Yes')
        sns.set_color_codes("muted")
        bar2 = sns.barplot(y="day", x="total_bill", data=smoker, color='b')

        # add legend
        top_bar = mpatches.Patch(color='darkblue', label='smoker = No')
        bottom_bar = mpatches.Patch(color='lightblue', label='smoker = Yes')
        plt.legend(handles=[top_bar, bottom_bar])
        plt.savefig('static/img/predicao.png')
        content_index['results'] = '/static/img/predicao.png'
    return render_template('index.html', **content_index)

@app.route('/modelos.html')
def modelos():
    return render_template('modelos.html', text = "")
@app.route('/Ranking.html')
def ranking():
    return render_template('ranking.html', text = "")
@app.route('/DataBase.html')
def database():
    return render_template('DataBase.html', text = "")
#API's
@app.route('/getrandomfake')
def getrandomfake_api():
    strJson = { "news": fn.getrandomfake() }
    return str(strJson)

@app.route('/getfakedatabase')
def getfakedatabase():
    return fn.getfakedatabase()

#if __name__=='__main__':
#    class daru():
#        pass


#@app.route('/resultado',methods=['POST','GET'])
#def result():
#    text = request.args.get("strFAKE")
#    #WordCloud().generate(text).to_file("./tepmlates/assets/img/wordcloud2.png")
#    return render_template('index.html',text = text)

#@app.route('/externo',methods=['POST','GET'])
#def predict():
#    return "<p> FAKE NEWS 30% </p> <hr> <p>VERDADEIRA 70%</p>"




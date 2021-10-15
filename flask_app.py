
from flask import Flask, render_template, request
import fakenewstools as fn
import modelos.modelo_01 as model
#from wordcloud import WordCloud

#app = Flask(__name__,template_folder="/home/fakenewsbr/mysite/templates")
app = Flask(__name__)#, template_folder = r"D:\Nuvem\ghdaru\OneDrive\030_DOUTORADO\460_MECAI\020_Probabilidade e Estatística\010_PROJETOFAKENEWS\fakenewsbr\templates")


content_index = {
    'text':'',
    'news':'',
    'database_size':fn.getfakequantity(),
    'vl_true':50,
    'vl_false':50
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
        current_text = "optPre: " + request.form.get('optPre','0')    
        current_text += " optFeature: " +  request.form.get('optFeature','0')    
        current_text += " optClassifier: " +  request.form.get('optClassifier','0')        
        content_index['vl_false'] = int(model.modelo(content_index['news'])*100)
        content_index['vl_true'] = 100 - content_index['vl_false']
    return render_template('index.html', **content_index)

@app.route('/algoritmos.html')
def algoritmos():
    return render_template('algoritmos.html', text = "")
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


#@app.route('/resultado',methods=['POST','GET'])
#def result():
#    text = request.args.get("strFAKE")
#    #WordCloud().generate(text).to_file("./tepmlates/assets/img/wordcloud2.png")
#    return render_template('index.html',text = text)

#@app.route('/externo',methods=['POST','GET'])
#def predict():
#    return "<p> FAKE NEWS 30% </p> <hr> <p>VERDADEIRA 70%</p>"



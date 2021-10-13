
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, render_template, request
#import pandas as pd
#from wordcloud import WordCloud

#app = Flask(__name__,template_folder="/home/fakenewsbr/mysite/templates")
app = Flask(__name__, template_folder = r"D:\Nuvem\ghdaru\OneDrive\030_DOUTORADO\460_MECAI\020_Probabilidade e Estatística\010_PROJETOFAKENEWS\fakenewsbr\templates")

@app.route('/index.html')
@app.route('/data')
def main():
    return render_template('index.html', text = "")
@app.route('/algoritmos.html')
def algoritmos():
    return render_template('algoritmos.html', text = "")
@app.route('/Ranking.html')
def ranking():
    return render_template('ranking.html', text = "")
@app.route('/DataBase.html')
def database():
    return render_template('DataBase.html', text = "")


#@app.route('/resultado',methods=['POST','GET'])
#def result():
#    text = request.args.get("strFAKE")
#    #WordCloud().generate(text).to_file("./tepmlates/assets/img/wordcloud2.png")
#    return render_template('index.html',text = text)

#@app.route('/externo',methods=['POST','GET'])
#def predict():
#    return "<p> FAKE NEWS 30% </p> <hr> <p>VERDADEIRA 70%</p>"



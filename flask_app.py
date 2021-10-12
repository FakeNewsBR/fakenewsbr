
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, render_template, request
import pandas as pd
from wordcloud import WordCloud

app = Flask(__name__,template_folder="/home/fakenewsbr/mysite/templates")



@app.route('/')
def main():
    return render_template('index.html', text = "")

@app.route('/resultado',methods=['POST','GET'])
def result():
    text = request.args.get("strFAKE")
    #WordCloud().generate(text).to_file("./tepmlates/assets/img/wordcloud2.png")
    return render_template('index.html',text = text)

@app.route('/externo',methods=['POST','GET'])
def predict():
    return "<p> FAKE NEWS 30% </p> <hr> <p>VERDADEIRA 70%</p>"



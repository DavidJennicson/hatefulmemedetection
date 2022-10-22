from flask import Flask,redirect,url_for,render_template,request
from pegasus import *
app = Flask(__name__)


@app.route('/',methods=["POST","GET"])
def hello_world():  # put application's code here
    if request.method=="POST":
        user=request.form["inp"]
        res=paraphrase(user)
        return render_template('res.html',content=res)
    return render_template('index.html')

@app.route('/results')
def show():
    return render_template('res.html')
if __name__ == '__main__':
    app.run()

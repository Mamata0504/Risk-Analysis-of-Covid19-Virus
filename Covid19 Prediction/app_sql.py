from flask import Flask,render_template,request,jsonify
import os,requests,pickle,sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
import Covid_Database_connect

app=Flask(__name__)

current_path=os.getcwd()
pickle_path = os.path.join(current_path, "assets", "Covid_Prediction.pkl")
model = pickle.load(open(pickle_path, "rb"))

@app.route("/")
@app.route("/home",methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    if request.method=='POST':
        sex=int(request.form['sex'])
        pneumonia=int(request.form['pneumonia'])
        age=int(request.form['age'])
        pregnancy=int(request.form['pregnancy'])
        diabetes=int(request.form['diabetes'])
        copd=int(request.form['copd'])
        asthma=int(request.form['asthma'])
        inmsupr=int(request.form['inmsupr'])
        hypertension=int(request.form['hypertension'])
        other_disease=int(request.form['other_disease'])
        cardiovascular=int(request.form['cardiovascular'])
        obesity=int(request.form['obesity'])
        renal_chronic=int(request.form['renal_chronic'])
        contact_other_covid=int(request.form['contact_other_covid'])
        x=[[sex,pneumonia,age,pregnancy,diabetes,copd,asthma,inmsupr,hypertension,other_disease,cardiovascular,obesity,renal_chronic,contact_other_covid]]
        sc=StandardScaler()
        x=sc.fit_transform(x)
        prediction=model.predict(x)
        output=int(prediction)
        Covid_Database_connect.insert_record(sex,pneumonia,age,pregnancy,diabetes,copd,asthma,inmsupr,hypertension,other_disease,cardiovascular,obesity,renal_chronic,contact_other_covid,output)
        if output==0:
            return render_template('index.html',prediction_text="Covid-19 Negative")
        else:
            return render_template('index.html',prediction_text="Covid-19 Positive")
    else:
        return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)
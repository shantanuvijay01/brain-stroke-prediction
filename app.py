from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#model = pickle.load(open("K_Neighbours_Classifier_model.pkl", "rb"))

@app.route('/', methods = ['GET'])
def Home():
    return render_template("index.html")

standard_to = StandardScaler()
@app.route('/prediction', methods = ["POST", "GET"])
def prediction():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    gender_encoded_Female = 0
    gender_encoded_Male = 0
    gender_encoded_Other = 0
    if(gender == 0):
        gender_encoded_Female = 1
    elif(gender == 1):
        gender_encoded_Male = 1
    elif(gender == 2):
        gender_encoded_Other = 1
    
    ever_married_Yes = 0
    ever_married_No = 0
    if(ever_married == 0):
        ever_married_No = 1
    elif(ever_married == 1):
        ever_married_Yes = 1
    
    Residence_type_Rural = 0
    Residence_type_Urban = 0

    if(Residence_type == 0):
        Residence_type_Urban = 1
    elif(Residence_type == 1):
        Residence_type_Rural = 1
    

    
    work_type_encoded_Govt_job = 0
    work_type_encoded_Never_worked = 0
    work_type_encoded_Private = 0
    work_type_encoded_self_employed = 0
    work_type_encoded_children = 0
    if(work_type == 0):
        work_type_encoded_Govt_job = 1
    elif(work_type == 1):
        work_type_encoded_Never_worked = 1
    elif(work_type == 2):
        work_type_encoded_Private = 1
    elif(work_type == 3):
        work_type_encoded_self_employed = 1
    elif(work_type == 4):
        work_type_encoded_children = 1
    
    smoking_status_encoded_Unknown = 0
    smoking_status_encoded_formerly_smoked = 0
    smoking_status_encoded_never_smoked = 0
    smoking_status_encoded_smokes = 0
    if(smoking_status == 0):
        smoking_status_encoded_Unknown = 1
    elif(smoking_status == 1):
        smoking_status_encoded_formerly_smoked = 1
    elif(smoking_status == 2):
        smoking_status_encoded_never_smoked = 1
    elif(smoking_status == 3):
        smoking_status_encoded_smokes = 1

    x = np.array([age,hypertension,heart_disease,
                  ever_married_Yes,ever_married_No,Residence_type_Rural,Residence_type_Urban,avg_glucose_level,
                  bmi,gender_encoded_Female,gender_encoded_Male,
                  gender_encoded_Other,work_type_encoded_Govt_job,work_type_encoded_Never_worked,
                  work_type_encoded_Private,work_type_encoded_self_employed,work_type_encoded_children,
                  smoking_status_encoded_Unknown,smoking_status_encoded_formerly_smoked,smoking_status_encoded_never_smoked,
                  smoking_status_encoded_smokes]).reshape(1, -1)

    #x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
    #              avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

    scaler_path = "scaler.pkl"
    scaler = None
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    x = scaler.transform(x)
    model_path = "knn.sav"
    dt = joblib.load(model_path)

    prediction = dt.predict(x)
    print(prediction)
    # for No Stroke Risk
    if prediction == 0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')

    
if __name__ == '__main__':
    app.run(debug=True)

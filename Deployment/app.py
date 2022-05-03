from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import pickle
import warnings
warnings.filterwarnings("ignore")
import flask
app = Flask(__name__)

def predict_X(X,model):
    
# Mode Imputation
    frequent_df = pd.read_csv('frequent_values.csv',index_col=0)
    for i in (X.columns.values):
        if (X[i].isnull().values.any()):
            X[i] = X[i].fillna(frequent_df.loc[i][0])
            
    X['temperature'] = int(X['temperature'])
    X['has_children'] = int(X['has_children'])
    X['toCoupon_GEQ15min'] = int(X['toCoupon_GEQ15min'])
    X['toCoupon_GEQ25min'] = int(X['toCoupon_GEQ25min'])
    X['direction_same'] = int(X['direction_same'])
            
# Feature Engineering
    if (list(X['toCoupon_GEQ15min'])[0] == 0):
        X['to_Coupon'] = 'within15min'
    elif (list(X['toCoupon_GEQ15min'])[0] == 1)and(list(X['toCoupon_GEQ25min'])[0] == 0):
        X['to_Coupon'] = 'within25min'
    else:
        X['to_Coupon'] = 'morethan25min'
        
    if (list(X['coupon'])[0] == 'Restaurant(<20)'):
        X['coupon_freq'] = (list(X['RestaurantLessThan20'])[0])
    elif (list(X['coupon'])[0] == 'Coffee House'):
        X['coupon_freq'] = (list(X['CoffeeHouse'])[0])
    elif (list(X['coupon'])[0] == 'Carry out & Take away'):
        X['coupon_freq'] = (list(X['CarryAway'])[0])
    elif (list(X['coupon'])[0] == 'Bar'):
        X['coupon_freq'] = (list(X['Bar'])[0])
    elif (list(X['coupon'])[0] == 'Restaurant(20-50)'):
        X['coupon_freq'] = (list(X['Restaurant20To50'])[0])

    occupation_dict = {'Healthcare Support':'High_Acceptance','Construction & Extraction':'High_Acceptance',
                       'Healthcare Practitioners & Technical':'High_Acceptance','Protective Service':'High_Acceptance',
                       'Architecture & Engineering':'High_Acceptance','Production Occupations':'Medium_High_Acceptance',
                       'Student':'Medium_High_Acceptance','Office & Administrative Support':'Medium_High_Acceptance',
                       'Transportation & Material Moving':'Medium_High_Acceptance',
                       'Building & Grounds Cleaning & Maintenance':'Medium_High_Acceptance','Management':'Medium_Acceptance',
                       'Food Preparation & Serving Related':'Medium_Acceptance','Life Physical Social Science':'Medium_Acceptance',
                       'Business & Financial':'Medium_Acceptance','Computer & Mathematical':'Medium_Acceptance',
                       'Sales & Related':'Medium_Low_Acceptance',
                       'Personal Care & Service':'Medium_Low_Acceptance','Unemployed':'Medium_Low_Acceptance',
                       'Farming Fishing & Forestry':'Medium_Low_Acceptance','Installation Maintenance & Repair':'Medium_Low_Acceptance',
                       'Education&Training&Library':'Low_Acceptance','Arts Design Entertainment Sports & Media':'Low_Acceptance',
                       'Community & Social Services':'Low_Acceptance','Legal':'Low_Acceptance','Retired':'Low_Acceptance'}
    X['occupation_class'] = X['occupation'].map(occupation_dict)
    X = X.drop(['occupation'], axis=1)
    
# Ordinal Encoding
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    X_Ordinal_encoding = vectorizer.transform(X)
    X_Ordinal_encoding = pd.DataFrame(X_Ordinal_encoding,columns=X.columns.values)
    
# Prediction
    y_pred = model.predict(X_Ordinal_encoding)
    y_pred_prob = model.predict_proba(X_Ordinal_encoding)
    return y_pred[0], y_pred_prob[0]

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open("model.pkl", "rb"))
    to_predict_dict = request.form.to_dict()
    to_predict_df = pd.DataFrame(to_predict_dict,index=[0])
    print(to_predict_df)
    prediction, prediction_prob = predict_X(to_predict_df,model)
    if (prediction==1):
        prob = round(prediction_prob[1],3)
    else:
        prob = round(prediction_prob[0],3)
    return flask.render_template('predict.html', prediction = prediction, prob = prob, coupon = to_predict_df['coupon'][0] )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    
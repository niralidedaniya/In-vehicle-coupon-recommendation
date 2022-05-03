from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import pickle
from prettytable import PrettyTable
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('/content/drive/MyDrive/Applied AI/CS1/in-vehicle-coupon-recommendation.csv')
# print(data.head(2))
print('## Reading Data Done')

X = data.drop(['Y'], axis=1)
y = data['Y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
X_train, X_set, y_train, y_set = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train)

X_train = X_train.drop(['car','direction_opp','toCoupon_GEQ5min'], axis=1)
X_test = X_test.drop(['car','direction_opp','toCoupon_GEQ5min'], axis=1)

print('\nX_train:',X_train.shape,'y_train:',y_train.shape)
print('X_test:',X_test.shape,'y_test:',y_test.shape)
print('\n## Data Cleaning & Train Test Split Done')

# Mode Imputation

# frequent_values = []
# for i in (X_train.columns.values):
#   frequent_values.append(X_train[i].value_counts().sort_values(ascending=False).index[0])
# frequent_df = pd.DataFrame(frequent_values,columns=['frequent_values'],index=X_train.columns.values)
# frequent_df.to_csv('/content/drive/MyDrive/Applied AI/CS1/Final/frequent_values.csv')

frequent_df = pd.read_csv('/content/drive/MyDrive/Applied AI/CS1/Final/frequent_values.csv',index_col=0)
X_train['Bar'] = X_train['Bar'].fillna(frequent_df.loc['Bar'][0])
X_train['CoffeeHouse'] = X_train['CoffeeHouse'].fillna(frequent_df.loc['CoffeeHouse'][0])
X_train['CarryAway'] = X_train['CarryAway'].fillna(frequent_df.loc['CarryAway'][0])
X_train['RestaurantLessThan20'] = X_train['RestaurantLessThan20'].fillna(frequent_df.loc['RestaurantLessThan20'][0])
X_train['Restaurant20To50'] = X_train['Restaurant20To50'].fillna(frequent_df.loc['Restaurant20To50'][0])

X_test['Bar'] = X_test['Bar'].fillna(frequent_df.loc['Bar'][0])
X_test['CoffeeHouse'] = X_test['CoffeeHouse'].fillna(frequent_df.loc['CoffeeHouse'][0])
X_test['CarryAway'] = X_test['CarryAway'].fillna(frequent_df.loc['CarryAway'][0])
X_test['RestaurantLessThan20'] = X_test['RestaurantLessThan20'].fillna(frequent_df.loc['RestaurantLessThan20'][0])
X_test['Restaurant20To50'] = X_test['Restaurant20To50'].fillna(frequent_df.loc['Restaurant20To50'][0])

print('\n## Mode Imputation Done')

# FE -- to_Coupon is combination of two features, toCoupon_GEQ15min and toCoupon_GEQ25min
to_Coupon_train = []; to_Coupon_test = []
for i in range(X_train.shape[0]):
    if (list(X_train['toCoupon_GEQ15min'])[i] == 0):
        to_Coupon_train.append('within15min')
    elif (list(X_train['toCoupon_GEQ15min'])[i] == 1)and(list(X_train['toCoupon_GEQ25min'])[i] == 0):
        to_Coupon_train.append('within25min')
    else:
        to_Coupon_train.append('morethan25min')
X_train['to_Coupon'] = to_Coupon_train
for i in range(X_test.shape[0]):
    if (list(X_test['toCoupon_GEQ15min'])[i] == 0):
        to_Coupon_test.append('within15min')
    elif (list(X_test['toCoupon_GEQ15min'])[i] == 1)and(list(X_test['toCoupon_GEQ25min'])[i] == 0):
        to_Coupon_test.append('within25min')
    else:
        to_Coupon_test.append('morethan25min')
X_test['to_Coupon'] = to_Coupon_test

# FE -- coupon_freq is combination of five features, RestaurantLessThan20, CoffeeHouse, CarryAway, Bar, Restaurant20To50
coupon_freq_train = []; coupon_freq_test = []
for i in range(X_train.shape[0]):
    if (list(X_train['coupon'])[i] == 'Restaurant(<20)'):
        coupon_freq_train.append(list(X_train['RestaurantLessThan20'])[i])
    elif (list(X_train['coupon'])[i] == 'Coffee House'):
        coupon_freq_train.append(list(X_train['CoffeeHouse'])[i])
    elif (list(X_train['coupon'])[i] == 'Carry out & Take away'):
        coupon_freq_train.append(list(X_train['CarryAway'])[i])
    elif (list(X_train['coupon'])[i] == 'Bar'):
        coupon_freq_train.append(list(X_train['Bar'])[i])
    elif (list(X_train['coupon'])[i] == 'Restaurant(20-50)'):
        coupon_freq_train.append(list(X_train['Restaurant20To50'])[i])       
X_train['coupon_freq'] = coupon_freq_train
for i in range(X_test.shape[0]):
    if (list(X_test['coupon'])[i] == 'Restaurant(<20)'):
        coupon_freq_test.append(list(X_test['RestaurantLessThan20'])[i])
    elif (list(X_test['coupon'])[i] == 'Coffee House'):
        coupon_freq_test.append(list(X_test['CoffeeHouse'])[i])
    elif (list(X_test['coupon'])[i] == 'Carry out & Take away'):
        coupon_freq_test.append(list(X_test['CarryAway'])[i])
    elif (list(X_test['coupon'])[i] == 'Bar'):
        coupon_freq_test.append(list(X_test['Bar'])[i])
    elif (list(X_test['coupon'])[i] == 'Restaurant(20-50)'):
        coupon_freq_test.append(list(X_test['Restaurant20To50'])[i])   
X_test['coupon_freq'] = coupon_freq_test

# occupation feature has 25 no of distinct values, which creates very sparsity in data after Encoding
# FE -- occupation_class where categorize all occupation in its suitable class.
occupation_dict = {'Healthcare Support':'High_Acceptance','Construction & Extraction':'High_Acceptance','Healthcare Practitioners & Technical':'High_Acceptance',
                   'Protective Service':'High_Acceptance','Architecture & Engineering':'High_Acceptance','Production Occupations':'Medium_High_Acceptance',
                    'Student':'Medium_High_Acceptance','Office & Administrative Support':'Medium_High_Acceptance','Transportation & Material Moving':'Medium_High_Acceptance',
                    'Building & Grounds Cleaning & Maintenance':'Medium_High_Acceptance','Management':'Medium_Acceptance','Food Preparation & Serving Related':'Medium_Acceptance',
                   'Life Physical Social Science':'Medium_Acceptance','Business & Financial':'Medium_Acceptance','Computer & Mathematical':'Medium_Acceptance',
                    'Sales & Related':'Medium_Low_Acceptance','Personal Care & Service':'Medium_Low_Acceptance','Unemployed':'Medium_Low_Acceptance',
                   'Farming Fishing & Forestry':'Medium_Low_Acceptance','Installation Maintenance & Repair':'Medium_Low_Acceptance','Education&Training&Library':'Low_Acceptance',
                    'Arts Design Entertainment Sports & Media':'Low_Acceptance','Community & Social Services':'Low_Acceptance','Legal':'Low_Acceptance','Retired':'Low_Acceptance'}
X_train['occupation_class'] = X_train['occupation'].map(occupation_dict)
X_test['occupation_class'] = X_test['occupation'].map(occupation_dict)

X_train = X_train.drop(['occupation'], axis=1)
X_test = X_test.drop(['occupation'], axis=1)

print('\n## Feature Engineering Done')

# Encoding
order = [['Work','Home','No Urgent Place'],['Kid(s)','Alone','Partner','Friend(s)'],['Rainy','Snowy','Sunny'],[30,55,80],['7AM','10AM','2PM','6PM','10PM'],
         ['Bar','Restaurant(20-50)','Coffee House','Restaurant(<20)','Carry out & Take away'],['2h','1d'],['Female','Male'],['below21','21','26','31','36','41','46','50plus'],
         ['Widowed','Divorced','Married partner','Unmarried partner','Single'],[0,1],
         ['Some High School','High School Graduate','Some college - no degree','Associates degree','Bachelors degree','Graduate degree (Masters or Doctorate)'],
         ['Less than $12500','$12500 - $24999','$25000 - $37499','$37500 - $49999','$50000 - $62499','$62500 - $74999','$75000 - $87499','$87500 - $99999','$100000 or More'],
         ['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],
         [0,1],[0,1],[0,1],['morethan25min','within25min','within15min'],['never','less1','1~3','4~8','gt8'],['Low_Acceptance','Medium_Low_Acceptance','Medium_Acceptance','Medium_High_Acceptance','High_Acceptance']]
Ordinal_enc = OrdinalEncoder(categories=order)
vectorizer = Ordinal_enc.fit(X_train)
# pickle.dump(vectorizer, open("/content/drive/MyDrive/Applied AI/CS1/Final/vectorizer.pkl", "wb"))

X_train_Ordinal_encoding = vectorizer.transform(X_train)
X_train_Ordinal_encoding = pd.DataFrame(X_train_Ordinal_encoding,columns=(X_train.columns.values)+'_OrE')

X_test_Ordinal_encoding = vectorizer.transform(X_test)
X_test_Ordinal_encoding = pd.DataFrame(X_test_Ordinal_encoding,columns=(X_test.columns.values)+'_OrE')

print('\nX_train_Ordinal_encoding:',X_train_Ordinal_encoding.shape)
print('X_test_Ordinal_encoding:',X_test_Ordinal_encoding.shape)

print('\n## Ordinal Encoding Done')

# Model
estimators = [
              ('GNB', GaussianNB()),
              ('LR', LogisticRegression(random_state=42,C=100)),
              ('KNN', KNeighborsClassifier(n_neighbors=21)),
              ('DT', DecisionTreeClassifier(class_weight='balanced', max_depth=10, min_samples_split=100, random_state=42)),
              ('LSVC',LinearSVC(C=1,random_state=42)),
              ('SVC', SVC(C=10,kernel='rbf',class_weight='balanced',probability=True)),
              ('GBC',GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=5, random_state=42)),
              ('RF',RandomForestClassifier(n_estimators=2000,criterion='gini',max_depth=20,max_features='log2',min_samples_leaf=3, random_state=42, n_jobs=-1)),
              ('HGB', HistGradientBoostingClassifier()),
              ('BC',BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=2000,random_state=42)),
              ('ABC',AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=1000,learning_rate=0.1,random_state=42)),
              ('ETC',ExtraTreesClassifier(n_estimators=500, random_state=42)),
              ('XGB', xgb.XGBClassifier(max_depth=10, n_estimators=500, random_state=42))
             ]

Stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
Stacking_model.fit(X_train_Ordinal_encoding, y_train)

print('\n## Model fit Done')

Train_loss = log_loss(y_train,Stacking_model.predict_proba(X_train_Ordinal_encoding))
Train_AUC = roc_auc_score(y_train,(Stacking_model.predict_proba(X_train_Ordinal_encoding))[:,1])
Test_loss = log_loss(y_test,Stacking_model.predict_proba(X_test_Ordinal_encoding))
Test_AUC = roc_auc_score(y_test,(Stacking_model.predict_proba(X_test_Ordinal_encoding))[:,1])

print('\nTrain_loss:',Train_loss)
print('Train_AUC:',Train_AUC)
print('Test_loss:',Test_loss)
print('Test_AUC:',Test_AUC)

pickle.dump(Stacking_model, open("/content/drive/MyDrive/Applied AI/CS1/Final/model.pkl", "wb"))
print('\n## Model dump Done')

X_set['Y'] = y_set
X_set.to_csv('/content/drive/MyDrive/Applied AI/CS1/Final/Test_Final.csv',index=False)

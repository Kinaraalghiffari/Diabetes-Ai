from django.views.generic import View
from django.shortcuts import render
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import io 
import urllib, base64
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import xgboost as xgb

def masterpages(request):
     return render(request,'masterpage.html')
     
def home(request):
     args = {}
     data = pd.read_csv('diabetes.csv')
     head1 = data.head()
     args['head'] = head1.to_html()
     descriptive = data.drop(columns = 'Outcome', axis=1).describe()
     args['desc'] =  descriptive.to_html()

     plt.figure(figsize=(7, 6))
     sns.distplot(data.Pregnancies[data.Outcome == 0], bins=6, color="r", label="Non Diabetes")
     sns.distplot(data.Pregnancies[data.Outcome == 1], bins=6, color="g", label="Diabetes")
     fig = plt.legend()
     buf = io.BytesIO()
     fig.figure.savefig(buf,format='png')
     buf.seek(0)
     string = base64.b64encode(buf.read())
     args['uri'] = urllib.parse.quote(string)

     plt.figure(figsize=(7, 6))
     sns.distplot(data.Glucose[data.Outcome == 0], bins=6, color="r", label="Non Diabetes")
     sns.distplot(data.Glucose[data.Outcome == 1], bins=6, color="g", label="Diabetes")
     fig = plt.legend()
     buf = io.BytesIO()
     fig.figure.savefig(buf,format='png')
     buf.seek(0)
     string = base64.b64encode(buf.read())
     args['uriglucose'] = urllib.parse.quote(string)

     plt.figure(figsize=(7, 6))
     sns.distplot(data.BloodPressure[data.Outcome == 0], bins=6, color="r", label="Non Diabetes")
     sns.distplot(data.BloodPressure[data.Outcome == 1], bins=6, color="g", label="Diabetes")
     fig = plt.legend()
     buf = io.BytesIO()
     fig.figure.savefig(buf,format='png')
     buf.seek(0)
     string = base64.b64encode(buf.read())
     args['uriBloodPressure'] = urllib.parse.quote(string)

     plt.figure(figsize=(7, 6))
     sns.distplot(data.SkinThickness[data.Outcome == 0], bins=6, color="r", label="Non Diabetes")
     sns.distplot(data.SkinThickness[data.Outcome == 1], bins=6, color="g", label="Diabetes")
     fig = plt.legend()
     buf = io.BytesIO()
     fig.figure.savefig(buf,format='png')
     buf.seek(0)
     string = base64.b64encode(buf.read())
     args['uriSkinThickness'] = urllib.parse.quote(string)

     plt.figure(figsize=(7, 6))
     sns.distplot(data.Insulin[data.Outcome == 0], bins=6, color="r", label="Non Diabetes")
     sns.distplot(data.Insulin[data.Outcome == 1], bins=6, color="g", label="Diabetes")
     fig = plt.legend()
     buf = io.BytesIO()
     fig.figure.savefig(buf,format='png')
     buf.seek(0)
     string = base64.b64encode(buf.read())
     args['uriInsulin'] = urllib.parse.quote(string)

     plt.figure(figsize=(7, 6))
     sns.distplot(data.BMI[data.Outcome == 0], bins=6, color="r", label="Non Diabetes")
     sns.distplot(data.BMI[data.Outcome == 1], bins=6, color="g", label="Diabetes")
     fig = plt.legend()
     buf = io.BytesIO()
     fig.figure.savefig(buf,format='png')
     buf.seek(0)
     string = base64.b64encode(buf.read())
     args['uriBMI'] = urllib.parse.quote(string)

     plt.figure(figsize=(7, 6))
     sns.distplot(data.DiabetesPedigreeFunction[data.Outcome == 0], bins=6, color="r", label="Non Diabetes")
     sns.distplot(data.DiabetesPedigreeFunction[data.Outcome == 1], bins=6, color="g", label="Diabetes")
     fig = plt.legend()
     buf = io.BytesIO()
     fig.figure.savefig(buf,format='png')
     buf.seek(0)
     string = base64.b64encode(buf.read())
     args['uriDiabetesPedigreeFunction'] = urllib.parse.quote(string)

     plt.figure(figsize=(7, 6))
     sns.distplot(data.Age[data.Outcome == 0], bins=6, color="r", label="Non Diabetes")
     sns.distplot(data.Age[data.Outcome == 1], bins=6, color="g", label="Diabetes")
     fig = plt.legend()
     buf = io.BytesIO()
     fig.figure.savefig(buf,format='png')
     buf.seek(0)
     string = base64.b64encode(buf.read())
     args['uriAge'] = urllib.parse.quote(string)

     return render(request,'home.html', args)

def detection(request):
     return render(request,'Detection.html')

def result(request): 
    args = {}
    data = pd.read_csv('diabetes.csv')
    data.groupby('Outcome').mean()
    features = data.drop(columns = 'Outcome', axis=1)
    labels = data['Outcome']
    scaler = StandardScaler()
    scaler.fit(features)
    standardized = scaler.transform(features)
    features = standardized
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=20)
    
    Pregnancies = request.POST['Pregnancies']
    Glucose = request.POST['Glucose']
    Pressure = request.POST['Pressure']
    Skin = request.POST['Skin']
    Insulin = request.POST['Insulin']
    BMI = request.POST['BMI']
    Pedigree = request.POST['pedigree']
    Age = request.POST['age']
    

#     SUPPORT VECTOR MACHINE
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    Training_prediction = classifier.predict(X_train)
    args['accuracy_svm_training'] = accuracy_score(Training_prediction, y_train)
    args['accuracy_training_svm_percentage'] =  round(args['accuracy_svm_training'],4) * 100
    Testing_prediction = classifier.predict(X_test)
    args['accuracy_svm_testing'] = accuracy_score(Testing_prediction, y_test)
    args['accuracy_testing_svm_percentage'] =  round(args['accuracy_svm_testing'],4) * 100
    cm = confusion_matrix(y_test, Testing_prediction)
    classes = ["Non Diabetes", "Diabetes"]
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    buf = io.BytesIO()
    cfm_plot.figure.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    args['uri'] = urllib.parse.quote(string)
    real_data = (Pregnancies,Glucose,Pressure,Skin,Insulin,BMI,Pedigree,Age)
    print(real_data)
    real_data_as_numpy_array = np.asarray(real_data)
    real_data_reshaped = real_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.transform(real_data_reshaped)
    print(std_data)
    prediction = classifier.predict(std_data)
    print(prediction)
    if (prediction[0] == 0):
        args['svm'] = "Potential non diabetic"
    else:
        args['svm'] = "Potential diabetic"

#     RANDOM FOREST CLASSIFIER
    RF = RandomForestClassifier(n_estimators = 10, random_state = 30)
    RF.fit(X_train, y_train)
    Training_prediction_RF = RF.predict(X_train)
    args['accuracy_training_rf'] = accuracy_score(Training_prediction_RF, y_train)
    args['accuracy_training_rf_percentage'] =  round(args['accuracy_training_rf'],3) * 100
    Testing_prediction_RF = RF.predict(X_test)
    args['accuracy_RF'] = accuracy_score(Testing_prediction_RF, y_test)
    args['accuracy_testing_rf_percentage'] =  round(args['accuracy_RF'],4) * 100
    cmrf = confusion_matrix(y_test, Testing_prediction_RF)
    classes = ["Non Diabetes", "Diabetes"]
    df_cfm = pd.DataFrame(cmrf, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plotrf= sn.heatmap(df_cfm, annot=True)
    buf = io.BytesIO()
    cfm_plotrf.figure.savefig(buf,format='png')
    buf.seek(0)
    stringrf = base64.b64encode(buf.read())
    args['uri_rf'] = urllib.parse.quote(stringrf)
    prediction = RF.predict(std_data)
    print(prediction)
    if (prediction[0] == 0):
        args['RF'] = "Potential non diabetic"
    else:
        args['RF'] = "Potential diabetic"

#     NAIVE BAYES CLASSIFIER
    NB_model = GaussianNB()
    NB_model.fit(X_train, y_train)
    Training_prediction_NB = NB_model.predict(X_train)
    args['accuracy_training_NB'] = accuracy_score(Training_prediction_NB, y_train)
    args['accuracy_training_nb_percentage'] =  round(args['accuracy_training_NB'],3) * 100
    Testing_prediction_NB = NB_model.predict(X_test)
    args['accuracy_testing_NB']= accuracy_score(Testing_prediction_NB, y_test)
    args['accuracy_testing_nb_percentage'] =  round(args['accuracy_testing_NB'],4) * 100
    cmnb = confusion_matrix(y_test, Testing_prediction_NB)
    classes = ["Non Diabetes", "Diabetes"]
    df_cfm = pd.DataFrame(cmnb, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plotnb= sn.heatmap(df_cfm, annot=True)
    buf = io.BytesIO()
    cfm_plotrf.figure.savefig(buf,format='png')
    buf.seek(0)
    stringnb = base64.b64encode(buf.read())
    args['uri_nb'] = urllib.parse.quote(stringnb)
    prediction = NB_model.predict(std_data)
    if (prediction[0] == 0):
        args['NB'] = "Potential non diabetic"
    else:
        args['NB'] = "Potential diabetic"

#     NAIVE BAYES CLASSIFIER
    model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_delta_step=0, max_depth=6, min_child_weight=0,
              monotone_constraints='()', n_estimators=100, n_jobs=8,
              num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
    model_xgb.fit(X_train, y_train)
    Training_prediction_xgb = model_xgb.predict(X_train)
    args['accuracy_training_xgb'] = accuracy_score(Training_prediction_xgb, y_train)
    args['accuracy_training_xgb_percentage'] =  round(args['accuracy_training_xgb'],4) * 100
    Testing_prediction_xgb = model_xgb.predict(X_test)
    args['accuracy_testing_xgb']= accuracy_score(Testing_prediction_xgb, y_test)
    args['accuracy_testing_xgb_percentage'] =  round(args['accuracy_testing_xgb'],4) * 100
    cmnb = confusion_matrix(y_test, Testing_prediction_xgb)
    classes = ["Non Diabetes", "Diabetes"]
    df_cfm = pd.DataFrame(cmnb, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plotnb= sn.heatmap(df_cfm, annot=True)
    buf = io.BytesIO()
    cfm_plotrf.figure.savefig(buf,format='png')
    buf.seek(0)
    stringnb = base64.b64encode(buf.read())
    args['uri_xgb'] = urllib.parse.quote(stringnb)
    prediction = model_xgb.predict(std_data)
    if (prediction[0] == 0):
        args['xgb'] = "Potential non diabetic"
    else:
        args['xgb'] = "Potential diabetic"
    



    return render(request,'result.html', args)
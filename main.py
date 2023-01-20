import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import tkinter as tk

app = tk.Tk()


def prep():
    df = pd.read_csv("train.csv")
    df.drop("id",inplace = True,axis = 1)
    df.replace(-1,np.NaN)
    df.dropna(inplace = True)
    X = df.drop('target',axis = 1)
    y = df["target"]
    smote = SMOTE(sampling_strategy='minority')
    xsm,ysm = smote.fit_resample(X,y)
    xtrain,xtest,ytrain,ytest  = train_test_split(xsm,ysm,test_size= 0.2,random_state=0)
    return xtrain,xtest,ytrain,ytest
def train_model(xtrain,xtest,ytrain,ytest):
    model = DecisionTreeClassifier()
    model.fit(xtrain,ytrain)
    return model
def predict_result(model,x):
    ypred = model.predict(x)
    return ypred

def ppp():
    print("preprosesing the data: -----------")
    xtrain,xtest,ytrain,ytest = prep()
    print('training the model ------------')
    model = train_model(xtrain,xtest,ytrain,ytest)
    s = inpt.get().split(',')
    x = list(map(float,s))
    x = np.array(x).reshape(1,-1)
    """
    exemple:

    6.0,3.0,6.0,1.0,4.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,8.0,1.0,0.0,0.0,0.8982067241641696,0.4,1.001265768207312,11.0,0.0,0.0,0.0,0.0,3.0,1.0,1.0,0.0,1.0,104.0,2.0,0.42404426556364117,1.017760993431469,0.3478340313290131,3.5821889147931434,0.0017932758358304435,0.8937235345745935,0.20448318958957612,2.0,0.0,8.0,3.0,9.0,2.0,9.0,2.0,0.0,3.0,9.0,0.0,0.0,1.0,0.0,1.0,0.0

    """
    print('predicting the result-------------')
    result = predict_result(model,x)
    print('result: ')
    resultl = tk.Label(app,text=str(result))
    resultl.pack()

app.title('Saber Souhaimi project')
app.geometry("800x400")
Label_one = tk.Label(app,text="enter you data here seperated by ','  please avoid writing spaces between words ")
inpt = tk.Entry()
qq = inpt.get()

submit_button = tk.Button(app,text = "SUBMIT",command=ppp)
Label_one.pack()
inpt.pack()
submit_button.pack()
app.mainloop()

print(qq)
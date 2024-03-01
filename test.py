from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample
import tkinter as tk 

model = load_model('best_model.h5')
train_df=pd.read_csv('Data/mitbih_train.csv',header=None)
train_df = train_df.iloc[:,:186].values

prediction = model.predict(train_df)
while True:
    heart_num = int(input())

    print(prediction[heart_num])
    maxi  = max(prediction[heart_num])
    x = 0
    for i in range(len(prediction[heart_num])):
        if prediction[heart_num][i] == maxi:
            x = i
            break
    types = ["Non-ecotic","Supraventricular ectopic","Ventricular ectopic","Fusion","unknown"]
    print("There is a ",maxi*100,"% Chance that it is ",types[i]," beats")
    print(len(train_df))

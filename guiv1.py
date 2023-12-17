from tkinter import *
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk) 
from tensorflow.keras.models import load_model
import pandas as pd
def ecg(heart_number):
    model = load_model('best_model.h5')
    train_df=pd.read_csv('Data/mitbih_train.csv',header=None)
    train_df = train_df.iloc[:,:186].values
    heart_num = heart_number
    prediction = model.predict(train_df)
    for i in range(len(prediction)):
        prediction[i] = (prediction[i]*100).round(2)
    
    print(prediction[heart_num])
    maxi  = max(prediction[heart_num])
    print(maxi)
    x = 0
    for i in range(len(prediction[heart_num])):
        if prediction[heart_num][i] == maxi:
            x = i
            break
    types = ["Non-ecotic","Supraventricular ectopic","Ventricular ectopic","Fusion","unknown"]
    ans = maxi,"% Chance that it is ",types[i]," beats"
    lbl = Label(window, text=ans,font=("Courier",9))
    lbl.pack(side = RIGHT)
    return train_df[heart_num],ans


def plot(canvas):
    train_df=pd.read_csv('Data/mitbih_train.csv',header=None)
    train_df = train_df.iloc[:,:186].values
    inp = text_box.get(1.0, "end-1c") 
    y,ans = ecg(int(inp))
    plot1 = fig.add_subplot()
    plot1.plot(y)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas,window)
    toolbar.update() 
    canvas.get_tk_widget().pack(side =LEFT)

window = Tk() 
window.title('Plotting in Tkinter') 
window.geometry("1280x800")
text_box = Text(window, height = 1, width = 52,font=200)
text_box.pack(side = LEFT)
fig = Figure(figsize = (10,100),dpi=100)
canvas = FigureCanvasTkAgg(fig,master = window)
plot_button = Button(master = window, command = plot(canvas), height = 2, width = 10, text = "Plot") 
plot_button.pack(side = RIGHT) 
window.mainloop() 

from subprocess import call
import tkinter as tk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


root = tk.Tk()
root.title("Worm Detection and Crop Recommendation System")



w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

image = Image.open('h1.jpg')

image = image.resize((700, 700), Image.LANCZOS)

background_image = ImageTk.PhotoImage(image)

background_image=ImageTk.PhotoImage(image)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)

#img=ImageTk.PhotoImage(Image.open("s1.jpg"))

#img2=ImageTk.PhotoImage(Image.open("s2.jpg"))

image = Image.open('h4.jpg')

image = image.resize((700, 700), Image.LANCZOS)

background_image = ImageTk.PhotoImage(image)

background_image=ImageTk.PhotoImage(image)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=680, y=0) #, relwidth=1, relheight=1)

#img=ImageTk.PhotoImage(Image.open("s1.jpg"))

#img2=ImageTk.PhotoImage(Image.open("s2.jpg"))

logo_label=tk.Label()
logo_label.place(x=0,y=0)

x = 1




  # , relwidth=1, relheight=1)
lbl = tk.Label(root, text="Welcome to Worm Detection and Soil Analysis for Crop Recommendation System", font=('times', 24,' bold underline '), height=1, width=75,bg="brown",fg="white")
lbl.place(x=0, y=0)
# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Model_Training():
    data = pd.read_csv("D:/23 Protech/100% code/worm detection and Soil anaylsis/23CP143-Worm+Crop classification/Crop_recommendation.csv")
    data.head()
    data = data.dropna()

   
    """Feature Selection => Manual"""
    x = data.drop(['label'], axis=1)
    data = data.dropna()
    print(type(x))
    
    y = data['label']
    print(type(y))
    x.shape
    

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=123)


    from sklearn.svm import SVC
    svcclassifier = SVC(kernel='linear')
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    label4 = tk.Label(root,text =str(repo),width=55,height=25,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=500,y=50)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as crop_rec.joblib",width=55,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=500,y=600)
    from joblib import dump
    dump (svcclassifier,"crop_rec.joblib")
    print("Model saved as crop_rec.joblib")


def call_file():
   from subprocess import call
   call(['python','Check.py'])


def worm():
   from subprocess import call
   call(['python','GUI_Master_old.py'])

def window():
    root.destroy()

# button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
#                     text="Data_Preprocessing", command=Data_Preprocessing, width=15, height=2)
# button2.place(x=5, y=120)

# button3 = tk.Button(root, foreground="white", background="purple", font=("times", 14, "bold"),
#                     text="Model Training", command=Model_Training, width=19, height=2)
# button3.place(x=10, y=200)

button4 = tk.Button(root, foreground="white", background="blue", font=("times", 20, "bold"),
                    text="Soil Analysis", command=call_file, width=19, height=1)
button4.place(x=500, y=280)

button4 = tk.Button(root, foreground="white", background="green", font=("times", 20, "bold"),
                    text="Worm Detection", command=worm, width=19, height=1)
button4.place(x=500, y=360)

exit = tk.Button(root, text="Exit", command=window, width=19, height=1, font=('times', 20, ' bold '),bg="red",fg="white")
exit.place(x=500, y=440)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
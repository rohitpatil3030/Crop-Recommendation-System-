from tkinter import *
def Train():
    """GUI"""
    import tkinter as tk
    import numpy as np
    import pandas as pd

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder

    root = tk.Tk()

    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (w, h))
    root.title("Crop_Recommendation_System")
    root.configure(background="#98FB98")
    
    N = tk.IntVar() 
    P = tk.IntVar()
    K = tk.IntVar()
    temperature = tk.DoubleVar()
    humidity = tk.DoubleVar()
    ph = tk.DoubleVar()
    rainfall = tk.DoubleVar()
    
    #===================================================================================================================
    def Detect():
        e1= N.get()
        print(e1)
        e2= P.get()
        print(e2)
        e3= K.get()
        print(e3)
        e4= temperature.get()
        print(e4)
        e5= humidity.get()
        print(e5)
        e6= ph.get()
        print(e6)
        e7= rainfall.get()
        print(e7)
        
        
        
        #########################################################################################
        
        from joblib import dump , load
        a1=load('D:/23 Protech/100% code/worm detection and Soil anaylsis/23CP143-Worm+Crop classification/crop_rec.joblib')
        v= a1.predict([[e1,e2,e3,e4,e5,e6,e7]])
        
        print(v)
        if v[0]==0:
            print("Maize")
            yes = tk.Label(root,text="This soil is suitable for crops like MAIZE",background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
            yes.place(x=300,y=600)
                     
        elif v[0]==1:
            print("rice")
            no = tk.Label(root, text="This soil is suitable for crops like RICE",background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
            no.place(x=300, y=600)
        
        elif v[0]==2:
            print("Chickpea")
            no = tk.Label(root, text="This soil is suitable for crops like CHICKPEA", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
            no.place(x=300, y=600) 
              
        elif v[0]==3:
              print("Kidneybeans")
              no = tk.Label(root, text="This soil is suitable for crops like KIDNEYBEANS", background="#98FB98", foreground="black",font=('times', 18, ' bold '),width=45)
              no.place(x=300, y=600)
                  
        elif v[0]==4:
             print("pigeonpeas")
             no = tk.Label(root, text="This soil is suitable for crops like PIGEONPEA", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
             no.place(x=300, y=600)
                 
        elif v[0]==5:
                    print("blackgrams")
                    no = tk.Label(root, text="This soil is suitable for crops like BLACKGRAMS", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
                    no.place(x=300, y=600)
                    
        elif v[0]==6:
                 print("lentil")
                 no = tk.Label(root, text="This soil is suitable for crops like LENTILS", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
                 no.place(x=300, y=600)
                 
        elif v[0]==7:
                 print("Pomegrante")
                 no = tk.Label(root, text="This soil is suitable for crops like POMEGRANTE", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
                 no.place(x=300, y=600)
                 
        elif v[0]==8:
                 print("Banana")
                 no = tk.Label(root, text="This soil is suitable for crops like BANANA", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
                 no.place(x=300, y=600)
                 
        elif v[0]==9:
                 print("Mango")
                 no = tk.Label(root, text="This soil is suitable for crops like MANGO", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
                 no.place(x=300, y=600)
                 
        elif v[0]==10:
                 print("Grapes")
                 no = tk.Label(root, text="This soil is suitable for crops like GRAPES", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
                 no.place(x=300, y=600)
                 
        elif v[0]==11:
                 print("Watermelon")
                 no = tk.Label(root, text="This soil is suitable for crops like WATERMELON", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
                 no.place(x=300, y=600)
                 
                 
        elif v[0]==12:
                 print("Muskmelon")
                 no = tk.Label(root, text="This soil is suitable for crops like MUSKMELON", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
                 no.place(x=300, y=600)
                 
        elif v[0]==13:
             print("Apple")
             no = tk.Label(root, text="This soil is suitable for crops like APPLE", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
             no.place(x=300, y=600)
                                                                                    
        elif v[0]==14:
              print("Orange")
              no = tk.Label(root, text="This soil is suitable for crops like ORANGE",background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
              no.place(x=300, y=600)
 
        elif v[0]==15:
              print("Papaya")
              no = tk.Label(root, text="This soil is suitable for crops like PAPAYA", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
              no.place(x=300, y=600)
              
        elif v[0]==16:
              print("Coconut")
              no = tk.Label(root, text="This soil is suitable for crops like COCONUT", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
              no.place(x=300, y=600)
              
        elif v[0]==17:
             print("Cotton")
             no = tk.Label(root, text="This soil is suitable for crops like COTTON", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
             no.place(x=300, y=600)  

        elif v[0]==18:
             print("Jute")
             no = tk.Label(root, text="This soil is suitable for crops like JUTE", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
             no.place(x=300, y=600)  
             
        elif v[0]==19:
               print("Coffee")
               no = tk.Label(root, text="This soil is suitable for crops like COFFEE", background="#98FB98", foreground="black",font=('times', 20, ' bold '),width=15)
               no.place(x=300, y=600) 
         
        else:
           print("No Crop")
           no = tk.Label(root, text="NO CROP PREDICTED", background="green", foreground="white",font=('times', 20, ' bold '),width=15)
           no.place(x=600, y=500)
        
                                                                                        
    label_l1 = tk.Label(root, text="Soil Analysis and Crop Recommendation System",font=("Times New Roman", 30, 'bold'),
                        background="#152238", fg="white", width=70, height=1)
    label_l1.place(x=0, y=0)                                                                                                                                                                                                                                                 

   
    l7=tk.Label(root,text="Ratio of Nitrogen content in soil",background="#98FB98",font=('times', 18, ' bold '),width=28)
    l7.place(x=150,y=100)
    N=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=N)
    N.place(x=600,y=100)    

    l1=tk.Label(root,text="Ratio of Phosphorous content in soil",background="#98FB98",font=('times', 18, ' bold '),width=28)
    l1.place(x=150,y=150)
    P=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=P)
    P.place(x=600,y=150)

    l2=tk.Label(root,text="Ratio of Potassium content in soil",background="#98FB98",font=('times', 18, ' bold '),width=28)
    l2.place(x=150,y=200)
    K=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=K)
    K.place(x=600,y=200)
    
    l3=tk.Label(root,text="Temperature in degree Celsius",background="#98FB98",font=('times', 18, ' bold '),width=28)
    l3.place(x=150,y=250)
    temperature=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=temperature)
    temperature.place(x=600,y=250)
    
    l6=tk.Label(root,text="Relative humidity in %",background="#98FB98",font=('times', 18, ' bold '),width=28)
    l6.place(x=150,y=300)
    temperature=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=humidity)
    temperature.place(x=600,y=300)
    
    l4=tk.Label(root,text="PH value of the soil",background="#98FB98",font=('times', 18, ' bold '),width=28)
    l4.place(x=150,y=350)
    ph=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=ph)
    ph.place(x=600,y=350)
    
    l5=tk.Label(root,text="Rainfall in mm",background="#98FB98",font=('times', 18, ' bold '),width=28)
    l5.place(x=150,y=400)
    rainfall=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=rainfall)
    rainfall.place(x=600,y=400)
   
   
    
    button1 = tk.Button(root,text="PREDICT",command=Detect,font=('times', 20, ' bold '),width=10,bg='#0B6623',fg='white')
    button1.place(x=500,y=500)

    


    root.mainloop()

Train()
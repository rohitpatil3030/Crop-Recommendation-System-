import tkinter as tk
from tkinter import ttk, LEFT, END
from tkinter import messagebox as ms
import sqlite3
from PIL import Image, ImageTk
#import re


##############################################+=============================================================
root = tk.Tk()
root.configure(background="white")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("700x500")
root.title("login form")




username = tk.StringVar()
password = tk.StringVar()
        

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
#image2 = Image.open('b1.jpg')
#image2 = image2.resize((w,h), Image.ANTIALIAS)

#background_image = ImageTk.PhotoImage(image2)

#background_label = tk.Label(root, image=background_image)

#background_label.image = background_image

#background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)







def registration():
    from subprocess import call
    call(["python","registration.py"])
    root.destroy()

def login():
        # Establish Connection

    with sqlite3.connect('evaluation.db') as db:
         c = db.cursor()

        # Find user If there is any take proper action
         db = sqlite3.connect('evaluation.db')
         cursor = db.cursor()
         cursor.execute("CREATE TABLE IF NOT EXISTS admin_registration"
                           "(Fullname TEXT, address TEXT, username TEXT, Email TEXT, Phoneno TEXT,Gender TEXT,age TEXT , password TEXT)")
         db.commit()
         find_entry = ('SELECT * FROM admin_registration WHERE username = ? and password = ?')
         c.execute(find_entry, [(username.get()), (password.get())])
         result = c.fetchall()

         if result:
            msg = ""
            # self.logf.pack_forget()
            # self.head['text'] = self.username.get() + '\n Loged In'
            # msg = self.head['text']
            #            self.head['pady'] = 150
            print(msg)
            ms.showinfo("messege", "User Verified")
            from subprocess import call
            call(['python','GUI_rec.py'])
            # =================================================
            root.destroy()

            

            # =================================================
         else:
           ms.showerror('Oops!', 'Username Or Password Did Not Found/Match.')


# frame_alpr = tk.LabelFrame(root, text=" --About us-- ", width=550, height=500, bd=5, font=('times', 14, ' bold '),bg="#7CCD7C")
# frame_alpr.grid(row=0, column=0, sticky='nw')
# frame_alpr.place(x=550, y=200)

# label_l2 = tk.Label(root, text="___ Login Form ___",font=("Times New Roman", 30, 'bold'),
#                     background="#EEEE00", fg="black", width=67, height=3)
# label_l2.place(x=0, y=90)


#bg1_icon=ImageTk.PhotoImage(file="m.jpg")

#bg_icon=ImageTk.PhotoImage(file="L.jpg")
user_icon=ImageTk.PhotoImage(file="l1.png")
pass_icon=ImageTk.PhotoImage(file="p1.jpg")
        
#bg_lbl=tk.Label(root,image=bg1_icon, width=800,height=600)
#bg_lbl.place(x=430,y=130)
        

        
Login_frame=tk.Frame(root,bg="white")
Login_frame.place(x=100,y=150)
        
#logolbl=tk.Label(Login_frame,image=bg_icon,bd=0).grid(row=0,columnspan=2,pady=20)
title=tk.Label(root, text="Login to System", font=("Times new roman", 30, "bold"),bd=3,fg="blue")
title.place(x=250,y=50,width=300)        
lbluser=tk.Label(Login_frame,text="Username :",image=user_icon,compound=LEFT,font=("Times new roman", 20, "bold"),bg="white",fg="black").grid(row=1,column=0,padx=20,pady=10)
txtuser=tk.Entry(Login_frame,bd=5,textvariable=username,font=("",15))
txtuser.grid(row=1,column=1,padx=20)
        
lblpass=tk.Label(Login_frame,text="Password :",image=pass_icon,compound=LEFT,font=("Times new roman", 20, "bold"),bg="white",fg="black").grid(row=2,column=0,padx=50,pady=10)
txtpass=tk.Entry(Login_frame,bd=5,textvariable=password,show="*",font=("",15))
txtpass.grid(row=2,column=1,padx=20)
        
btn_log=tk.Button(Login_frame,text="Submit",command=login,width=15,font=("Times new roman", 14, "bold"),bg="blue",fg="white",borderwidth = 0)
btn_log.grid(row=3,column=1,pady=10)
btn_reg=tk.Button(Login_frame,text="Sign Up",command=registration,width=15,font=("Times new roman", 14, "bold"),bg="green",fg="white",borderwidth = 0)
btn_reg.grid(row=3,column=0,pady=10)
       
        
    
       
        # Login Function




    
def window():
  root.destroy()
  
  
    
    


button4 = tk.Button(root, text="EXIT", command=window, width=10, height=1,font=('times 15 bold underline'),bd=0,bg="red", fg="white")
button4.place(x=1200, y=600)


# button1 = tk.Button(frame_alpr, text="Login", command=log, width=14, height=1,font=('times', 20, ' bold '), bg="Black", fg="white")
# button1.place(x=150, y=110)

# button2 = tk.Button(frame_alpr, text="Register",command=reg,width=14, height=1,font=('times', 20, ' bold '), bg="black", fg="white")
# button2.place(x=150, y=200)

# button3 = tk.Button(frame_alpr, text="Exit",command=window,width=14, height=1,font=('times', 20, ' bold '), bg="red", fg="white")
# button3.place(x=150, y=300)





root.mainloop()
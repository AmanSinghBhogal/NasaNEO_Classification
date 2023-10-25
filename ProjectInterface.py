from tkinter import *
# from ClassicalGausianNB import cnb

# Initializing Tkinter application
app = Tk()

# Giving the app a title:
app.title("Nasa Nearest Earth Object Detection")

# Giving the application dimensions:
app.geometry("1450x770")
app.resizable(FALSE, FALSE)
app.config(bg='black')

appTitle1 = Label(app, text="Capstone Project", font=("MV Boli",18,"bold"),fg="white", bg="black",justify="center")
appTitle1.place(x=620, y=10)
appTitle2 = Label(app, text="By", font=("MV Boli",16),fg="white", bg="black")
appTitle2.place(x=720, y=40)
appTitle3 = Label(app, text="Aman Singh Bhogal and Mausmi Sinha", font=("MV Boli",16),fg="white", bg="black")
appTitle3.place(x=550, y=70)

# For Classical Gaussian Naive nayes
appCNB1 = Label(app, text="Classical Gaussian Naive Bayes", font=("MV Boli",16),fg="white", bg="black")
appCNB1.place(x=250, y=150)

appCNB2 = Label(app, text="Enter Estimated Diameter : ", font=("MV Boli",16),fg="white", bg="black")
appCNB2.place(x=120, y=240)

box1 = Entry(app, font=("MV Boli",25),bd="2")
box1.place(x=420, y=240,width=250,height=30) 
box1.focus_set()

# For Quantum Naive bayes
appQNB1 = Label(app, text="Quantum Naive Bayes", font=("MV Boli",16),fg="white", bg="black")
appQNB1.place(x=950, y=150)
# # For taking entry from user
# box1 = Entry(app, font=("Arial",25),bd="2")
# box1.place(x=100, y=90,width=250,height=30) 
# box1.focus_set()
app.mainloop()
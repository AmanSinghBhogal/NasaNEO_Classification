from tkinter import *
# from ClassicalGausianNB import cnb

# Initializing Tkinter application
app = Tk()

# Giving the app a title:
app.title("Nasa Nearest Earth Object Detection")

# Giving the application dimensions:
AppWidth = 1450
AppHeight = 770
app.geometry("1450x770")
app.resizable(FALSE, FALSE)
app.config(bg='black')

# Coordinates for rows:
Row0 = 150
Row1 = 210
Row2 = 270
Row3 = 330
Row4 = 390
Row5 = 450

Col1 = 40
Col2 = 370

appTitle1 = Label(app, text="Capstone Project", font=("MV Boli",18,"bold"),fg="white", bg="black", justify="center")
appTitle1.place(x=0, y=0, width=AppWidth)
appTitle2 = Label(app, text="By", font=("MV Boli",16),fg="white", bg="black",justify="center")
appTitle2.place(x=0, y=40, width=AppWidth)
appTitle3 = Label(app, text="Aman Singh Bhogal and Mausmi Sinha", font=("MV Boli",16),fg="white", bg="black", justify="center")
appTitle3.place(x=0, y=70, width=AppWidth)

# For Classical Gaussian Naive nayes
appCNB1 = Label(app, text="Classical Gaussian Naive Bayes", font=("MV Boli",16),fg="white", bg="black")
appCNB1.place(x=180, y=Row0)

# For Quantum Naive bayes
appQNB1 = Label(app, text="Quantum Naive Bayes", font=("MV Boli",16),fg="white", bg="black")
appQNB1.place(x=850, y=Row0)

# Input for Gaussian Naive Bayes:
appCNB2 = Label(app, text="Enter Estimated Diameter Min: ", font=("MV Boli",16),fg="white", bg="black")
appCNB2.place(x=Col1, y=Row1)
DiaMin = Entry(app, font=("MV Boli",14),bd="2", borderwidth=5, justify="right", relief=FLAT)
DiaMin.place(x=Col2, y=Row1, width=250,height=30) 
DiaMin.focus_set()

appCNB3 = Label(app, text="Enter Estimated Diameter Max: ", font=("MV Boli",16),fg="white", bg="black")
appCNB3.place(x=Col1, y=Row2)
DiaMax = Entry(app, font=("MV Boli",14),bd="2", borderwidth=5, justify="right", relief=FLAT)
DiaMax.place(x=Col2, y=Row2, width=250,height=30) 

appCNB4 = Label(app, text="Enter Relative Velocity: ", font=("MV Boli",16),fg="white", bg="black")
appCNB4.place(x=Col1, y=Row3)
RelVel = Entry(app, font=("MV Boli",14),bd="2", borderwidth=5, justify="right", relief=FLAT)
RelVel.place(x=Col2, y=Row3, width=250,height=30) 

appCNB5 = Label(app, text="Enter Miss Distance: ", font=("MV Boli",16),fg="white", bg="black")
appCNB5.place(x=Col1, y=Row4)
Misdis = Entry(app, font=("MV Boli",14),bd="2", borderwidth=5, justify="right", relief=FLAT)
Misdis.place(x=Col2, y=Row4, width=250,height=30) 

appCNB6 = Label(app, text="Enter Absolute Magnitude: ", font=("MV Boli",16),fg="white", bg="black")
appCNB6.place(x=Col1, y=Row5)
AbsoMag = Entry(app, font=("MV Boli",14),bd="2", borderwidth=5, justify="right", relief=FLAT)
AbsoMag.place(x=Col2, y=Row5, width=250,height=30) 

app.mainloop()
from tkinter import *
from ClassicalGausianNB_runner import cnb

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
Row6 = 540
Row7 = 600

Col1 = 40
Col2 = 370
Col3 = 650
Col4 = 980

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

displayValue = StringVar()
displayValue.set('')
CResLab = Label(app, text="Classical Hazardous = ", font=("MV Boli",16),fg="white", bg="black")
CResLab.place(x=Col1+100,y=Row7)
CResult = Label(app, textvariable=displayValue, font=("MV Boli",16),fg="white", bg="black", bd=2)
CResult.place(x=Col1+350,y=Row7)

# Classical Gaussian NB Call:
def hitCNB():
    displayValue.set(str(bool(cnb([float(DiaMin.get()), float(DiaMax.get()), float(RelVel.get()), float(Misdis.get()), float(AbsoMag.get())]))))

appCNB7 = Button(app, text="Submit", command=hitCNB, font=("MV Boli",16),justify="center",fg="white", bg="black", highlightcolor="grey")
appCNB7.place(x=Col1+250, y=Row6)


# Input for Quantum Bayesian Interference:
appQNB1 = Label(app, text="Enter Estimated Max Diameter: ", font=("MV Boli",16),fg="white", bg="black")
appQNB1.place(x=Col3, y=Row1)
QMaxDia = Entry(app, font=("MV Boli",14),bd="2", borderwidth=5, justify="right", relief=FLAT)
QMaxDia.place(x=Col4, y=Row1, width=250,height=30) 

appQNB2 = Label(app, text="Enter Relative Velocity: ", font=("MV Boli",16),fg="white", bg="black")
appQNB2.place(x=Col3, y=Row2)
QRelVel = Entry(app, font=("MV Boli",14),bd="2", borderwidth=5, justify="right", relief=FLAT)
QRelVel.place(x=Col4, y=Row2, width=250,height=30) 

appQNB3 = Label(app, text="Enter Miss Distance: ", font=("MV Boli",16),fg="white", bg="black")
appQNB3.place(x=Col3, y=Row3)
QMisDis = Entry(app, font=("MV Boli",14),bd="2", borderwidth=5, justify="right", relief=FLAT)
QMisDis.place(x=Col4, y=Row3, width=250,height=30) 

QdisplayValue = StringVar()
QdisplayValue.set('')
QResLab = Label(app, text="Quantum Hazardous = ", font=("MV Boli",16),fg="white", bg="black")
QResLab.place(x=Col3+100,y=Row7)
QResult = Label(app, textvariable=QdisplayValue, font=("MV Boli",16),fg="white", bg="black", bd=2)
QResult.place(x=Col3+350,y=Row7)

# Quantum NB Call:
def hitQNB():
    QdisplayValue.set(str(True))

appQNB7 = Button(app, text="Submit", command=hitQNB, font=("MV Boli",16),justify="center",fg="white", bg="black", highlightcolor="grey")
appQNB7.place(x=Col3+250, y=Row6)

# Quantum NB Call:
# Listing Initialize the parameters
# Step 0: Initialize the parameter values
params = {
    'p_norm_large_slow_hazardous': 0.45,
    'p_norm_large_slow_nonhazardous': 0.46,
    'p_norm_large_fast_hazardous': 0.47,
    'p_norm_large_fast_nonhazardous': 0.48,
    'p_norm_small_slow_hazardous': 0.49,
    'p_norm_small_slow_nonhazardous': 0.51,
    'p_norm_small_fast_hazardous': 0.52,
    'p_norm_small_fast_nonhazardous': 0.53,
}


app.mainloop()
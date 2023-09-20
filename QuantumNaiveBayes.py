import pandas as pd
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, execute, Aer
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import asin, sqrt, pi, log
from statistics import mean
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Loaded the Dataset
data = pd.read_csv('neo_v2.csv')

# Writing code for Pre-Procesing

cat_max_dia = []
cat_RV = []
cat_miss = []

max_dia_mean = mean(data['est_diameter_max'])
Rv_mean = mean(data['relative_velocity'])
miss_mean = mean(data['miss_distance'])

print("\n\nThe Mean Max Diameter is: ", max_dia_mean)
print("The Mean Relative Velocity is: ", Rv_mean)
print("The Mean Miss Distance is: ", miss_mean)

for i,j, z in zip(data['est_diameter_max'], data['relative_velocity'], data['miss_distance']):
    if i>=max_dia_mean:
        cat_max_dia.append("Large")
    else :
        cat_max_dia.append("Small")
        
    if j>=Rv_mean:
        cat_RV.append("Fast") 
    else:
        cat_RV.append("Slow")

    if z>=miss_mean:
        cat_miss.append("More")  
    else:
        cat_miss.append("Less")

processed_data = pd.DataFrame(list(zip(data['est_diameter_max'], data['relative_velocity'],data['miss_distance'], cat_max_dia, cat_RV,cat_miss, data['hazardous'])),columns=['Max_Diameter','Relative_Velocity','Miss_Distance', 'Categorized_Diameter', 'Categorized_Relative_Vel','Categorised_Miss_Distance','Hazardous'])

# Saving the Processed Data to get a better view
processed_data.to_csv('processedData.csv')
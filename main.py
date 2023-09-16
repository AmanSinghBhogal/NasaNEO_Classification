import pandas as pd
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Loaded the Dataset
data = pd.read_csv  ('neo_v2.csv')

# Writing code for Pre-Procesing

cat_max_dia = []
cat_RV = []
cat_miss = []

for i,j, z in zip(data['est_diameter_max'], data['relative_velocity'], data['miss_distance']):
    if i>=0 and i<0.25:
        cat_max_dia.append("Very Small")
    elif i>=0.25 and i<0.5:
        cat_max_dia.append("Small")
    elif i>=0.5 and i<0.75:
        cat_max_dia.append("Medium")
    elif i>=0.75 and i<1.0:
        cat_max_dia.append("Large")
    else :
        cat_max_dia.append("Very Large")
        
    if j>=0 and j<25000:
        cat_RV.append("Very Slow")  
    elif j>= 25000 and j<50000 :
        cat_RV.append("Slow")
    elif j>= 50000 and j<75000 :
        cat_RV.append("Medium")
    elif j>= 75000 and j<100000 :
        cat_RV.append("Fast")
    else:
        cat_RV.append("Fast as fuck")

    if z>=0 and z<10000000:
        cat_miss.append("Very Less")  
    elif z>= 10000000 and z<20000000 :
        cat_miss.append("Less")
    elif z>= 20000000 and z<30000000 :
        cat_miss.append("Medium")
    elif z>= 30000000 and z<40000000 :
        cat_miss.append("Bohot")
    else:
        cat_miss.append("Bohot jyda")

    
processed_data = pd.DataFrame(list(zip(data['est_diameter_max'], data['relative_velocity'],data['miss_distance'], cat_max_dia, cat_RV,cat_miss, data['hazardous'])),columns=['Max Diameter','Relative Velocity','Miss Distance', 'Categorized Diameter', 'Categorized Relatice Vel','Categorised Miss Distance','Hazardous'])

# Saving the Processed Data to get a better view
processed_data.to_csv('processedData.csv')



# Splitted the Dataset into Training and Testing
train_input, test_input = train_test_split(processed_data, test_size=0.3, random_state=0)

# Printing the Number of True and False Records in Train and Test Dataset
print("Train Dataset: No of True: {}, No. False: {}".format(len(train_input[train_input['Hazardous'] == True]), len(train_input[train_input['Hazardous'] == False])))
print("Test Dataset: No of True: {}, No. False: {}".format(len(test_input[test_input['Hazardous'] == True]), len(test_input[test_input['Hazardous'] == False])))

# # Printing Number of records in training and testing
print("There are {} Training Records and {} Testing Records".format(train_input.shape[0], test_input.shape[0]))

# need to write preprocess function for calculating backward probability

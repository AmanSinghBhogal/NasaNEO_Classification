import pandas as pd
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, execute, Aer
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import asin, sqrt
from math import log
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

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
        cat_RV.append("Very Fast")

    if z>=0 and z<10000000:
        cat_miss.append("Very Less")  
    elif z>= 10000000 and z<20000000 :
        cat_miss.append("Less")
    elif z>= 20000000 and z<30000000 :
        cat_miss.append("Medium")
    elif z>= 30000000 and z<40000000 :
        cat_miss.append("More")
    else:
        cat_miss.append("Too much")

    
processed_data = pd.DataFrame(list(zip(data['est_diameter_max'], data['relative_velocity'],data['miss_distance'], cat_max_dia, cat_RV,cat_miss, data['hazardous'])),columns=['Max_Diameter','Relative_Velocity','Miss_Distance', 'Categorized_Diameter', 'Categorized_Relative_Vel','Categorised_Miss_Distance','Hazardous'])

# Saving the Processed Data to get a better view
processed_data.to_csv('processedData.csv')



# Splitted the Dataset into Training and Testing
train_input, test_input = train_test_split(processed_data, test_size=0.3, random_state=0)

# Printing Number of records in training and testing
print("There are {} Training Records and {} Testing Records".format(train_input.shape[0], test_input.shape[0]))

# Printing the Number of True and False Records in Train and Test Dataset
print("Train Dataset: No of True: {}, No. False: {}".format(len(train_input[train_input['Hazardous'] == True]), len(train_input[train_input['Hazardous'] == False])))
print("Test Dataset: No of True: {}, No. False: {}".format(len(test_input[test_input['Hazardous'] == True]), len(test_input[test_input['Hazardous'] == False])))


# need to write preprocess function for calculating backward probability

# Function for Calculating Category Prob. :
def prob_hazard_calc(df, category_name, category_val):
    pop = df[df[category_name] == category_val]
    hazard_pop = pop[pop['Hazardous'] == True]
    p_pop = len(hazard_pop)/len(pop)
    return p_pop, len(pop)

# for very small:
p_vsmall, pop_vsmall = prob_hazard_calc(train_input, "Categorized_Diameter", "Very Small")
#print(p_vsmall, pop_vsmall)

# for small:
p_small, pop_small=  prob_hazard_calc(train_input, "Categorized_Diameter", "Small")
# print(p_small)

# for medium:
p_med, pop_med = prob_hazard_calc(train_input, "Categorized_Diameter", "Medium")
# print(p_med)

# for Large:
p_large, pop_large = prob_hazard_calc(train_input, "Categorized_Diameter", "Large")
# print(p_large)

# for Very Large:
p_vlarge, pop_vlarge = prob_hazard_calc(train_input, "Categorized_Diameter", "Very Large")
# print(p_vlarge)

print("\n\nPrinting Chances of Max Diameter Objects given they are hazardous:\n")
print("{} Very Small Diameter objects had {} chances of being hazardous".format(pop_vsmall, p_vsmall))
print("{} Small Diameter objects had {} chances of being hazardous".format(pop_small, p_small))
print("{} Medium Diameter objects had {} chances of being hazardous".format(pop_med, p_med))
print("{} Large Diameter objects had {} chances of being hazardous".format(pop_large, p_large))
print("{} Very Large Diameter objects had {} chances of being hazardous".format(pop_vlarge, p_vlarge))
# # For Relative Velocity:

# for Very Slow:
p_vslow, pop_vslow = prob_hazard_calc(train_input, "Categorized_Relative_Vel", "Very Slow")
# print(p_vslow)

# for Slow:
p_slow, pop_slow = prob_hazard_calc(train_input, "Categorized_Relative_Vel", "Slow")
# print(p_slow)

# for Medium:
p_Rmed, pop_Rmed = prob_hazard_calc(train_input, "Categorized_Relative_Vel", "Medium")
# print(p_med)

# for Fast:
p_fast, pop_fast = prob_hazard_calc(train_input, "Categorized_Relative_Vel", "Fast")
# print(p_fast)

# for Very Fast:
p_vfast, pop_vfast = prob_hazard_calc(train_input, "Categorized_Relative_Vel", "Very Fast")
# print(p_vfast)

print("\n\nPrinting Chances of Relative Velocity Objects given they are hazardous:\n")
print("{} Very Slow Relative Velocity objects had {} chances of being hazardous".format(pop_vslow, p_vslow))
print("{} Slow Relative Velocity objects had {} chances of being hazardous".format(pop_slow, p_slow))
print("{} Medium Relative Velocity objects had {} chances of being hazardous".format(pop_Rmed, p_Rmed))
print("{} Fast Relative Velocity objects had {} chances of being hazardous".format(pop_fast, p_fast))
print("{} Very Fast Relative Velocity objects had {} chances of being hazardous".format(pop_vfast, p_vfast))

# Specifying the marginal probability
def prob_to_angle(prob):
    """
    Converts a given P(psi) value into an equivalent theta value.
    """
    return 2*asin(sqrt(prob))

qc = QuantumCircuit(1)

# Set qubit to prior since in training data number of hazardous objects = 6183 and total records = 63585 therefore, P(hazardous) = 6183/63585 = 0.097
qc.ry(prob_to_angle(0.097), 0)

# execute the qc
results = execute(qc,Aer.get_backend('statevector_simulator')).result().get_counts()
plot_histogram(results)




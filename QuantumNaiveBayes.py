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

# Splitted the Dataset into Training and Testing
train_input, test_input = train_test_split(processed_data, test_size=0.3, random_state=0)

# Printing Number of records in training and testing
print("There are {} Training Records and {} Testing Records".format(train_input.shape[0], test_input.shape[0]))

# Printing the Number of True and False Records in Train and Test Dataset
print("Train Dataset: No of True: {}, No. False: {}".format(len(train_input[train_input['Hazardous'] == True]), len(train_input[train_input['Hazardous'] == False])))
print("Test Dataset: No of True: {}, No. False: {}".format(len(test_input[test_input['Hazardous'] == True]), len(test_input[test_input['Hazardous'] == False])))

# need to write preprocess function for calculating backward probability

# Function for Calculating Category Prob. 
def prob_hazard_calc(df, category_name, category_val):
    pop = df[df[category_name] == category_val]
    hazard_pop = pop[pop['Hazardous'] == True]
    p_pop = len(hazard_pop)/len(pop)
    return p_pop, len(pop)

# For Max Diameter

# for small:
p_small, pop_small = prob_hazard_calc(train_input, "Categorized_Diameter", "Small")
# print(p_small, pop_small)

# for Large:
p_large, pop_large = prob_hazard_calc(train_input, "Categorized_Diameter", "Large")
# print(p_large, pop_large)

print("\n\nPrinting Chances of Max Diameter Objects given they are hazardous:\n")
print("{} Small Diameter objects had {} chances of being hazardous".format(pop_small, p_small))
print("{} Large Diameter objects had {} chances of being hazardous".format(pop_large, p_large))

# For Relative Velocity:

# for Slow:
p_slow, pop_slow = prob_hazard_calc(train_input, "Categorized_Relative_Vel", "Slow")
# print(p_slow, pop_slow)

# for Fast:
p_fast, pop_fast = prob_hazard_calc(train_input, "Categorized_Relative_Vel", "Fast")
# print(p_fast, pop_fast)

print("\n\nPrinting Chances of Relative Velocity Objects given they are hazardous:\n")
print("{} Slow Relative Velocity objects had {} chances of being hazardous".format(pop_slow, p_slow))
print("{} Fast Relative Velocity objects had {} chances of being hazardous".format(pop_fast, p_fast))

# For Miss Distance:

# For Less
p_less, pop_less = prob_hazard_calc(train_input, "Categorised_Miss_Distance", "Less")
# print(p_less, pop_less)

# For More:
p_more, pop_more = prob_hazard_calc(train_input, "Categorised_Miss_Distance", "More")
# print(p_more, pop_more)

print("\n\nPrinting Chances of Miss Distance Objects given they are hazardous:\n")
print("{} Less Miss Distance objects had {} chances of being hazardous".format(pop_less, p_less))
print("{} More Miss Distance objects had {} chances of being hazardous".format(pop_more, p_more))

# Implementing Quantum Naive Bayes Circuit:

# Specifying the marginal probability
def prob_to_angle(prob):
    """
    Converts a given P(psi) value into an equivalent theta value.
    """
    return 2*asin(sqrt(prob))

# Initialize the quantum circuit
qc = QuantumCircuit(3)

# Set qubit0 to p_small i.e for Max Diameter
qc.ry(prob_to_angle(p_large), 0)

# Set qubit1 to p_fast i.e for Relative Velocity
qc.ry(prob_to_angle(p_fast), 1)

# Defining the CCRY‐gate:
def ccry(qc, theta, control1, control2, controlled):
    qc.cry(theta/2, control2, controlled)
    qc.cx(control1, control2)
    qc.cry(-theta/2, control2, controlled)
    qc.cx(control1, control2)
    qc.cry(theta/2, control1, controlled)

# Calculating the conditional probabilities

# fast Relative Velocity and large Diameter
population_fast=train_input[train_input.Categorized_Relative_Vel.eq("Fast")]
population_fast_large= population_fast[population_fast.Categorized_Diameter.eq("Large")]
hazardous_fast_large = population_fast_large[population_fast_large.Hazardous.eq(1)]
p_hazardous_fast_large=len(hazardous_fast_large)/len(population_fast_large)

# fast Relative Velocity and small Diameter
population_fast_small = population_fast[population_fast.Categorized_Diameter.eq("Small")]
hazardous_fast_small = population_fast_small[population_fast_small.Hazardous.eq(1)]
p_hazardous_fast_small=len(hazardous_fast_small)/len(population_fast_small)

# Slow Relative Velocity and Large Diameter
population_slow = train_input[train_input.Categorized_Relative_Vel.eq("Slow")]
population_slow_large = population_slow[population_slow.Categorized_Diameter.eq("Large")]
hazardous_slow_large=population_slow_large[population_slow_large.Hazardous.eq(1)]
p_hazardous_slow_large=len(hazardous_slow_large)/len(population_slow_large)

# Slow Relative Velocity and Small Diameter
population_slow_small = population_slow[population_slow.Categorized_Diameter.eq("Small")]
hazardous_slow_small = population_slow_small[population_slow_small.Hazardous.eq(1)]
p_hazardous_slow_small=len(hazardous_slow_small)/len(population_slow_small)

# Initializing the child node:

# set state |00> to conditional probability of slow RV and small Diameter
qc.x(0)
qc.x(1)
ccry(qc,prob_to_angle(p_hazardous_slow_small),0,1,2)
qc.x(0)
qc.x(1)

# set state |01> to conditional probability of slow RV and large Diameter
qc.x(0)
ccry(qc,prob_to_angle(p_hazardous_slow_large),0,1,2)
qc.x(0)

# set state |10> to conditional probability of fast RV and small Diameter
qc.x(1)
ccry(qc,prob_to_angle(p_hazardous_fast_small),0,1,2)
qc.x(1)

# set state |11> to conditional probability of fast RV and large Diameter
ccry(qc,prob_to_angle(p_hazardous_fast_large),0,1,2)

# Circuit execution

# execute the qc
results = execute(qc,Aer.get_backend('statevector_simulator')).result().get_counts()
plot_histogram(results)

# Quantum circuit with classical register
qr = QuantumRegister(3)
cr = ClassicalRegister(1)
qc = QuantumCircuit(qr, cr)

# Listing Run the circuit including a measurement

# -- INCLUDE ALL GATES HERE --
# Set qubit0 to p_small i.e for Max Diameter
qc.ry(prob_to_angle(p_large), 0)

# Set qubit1 to p_fast i.e for Relative Velocity
qc.ry(prob_to_angle(p_fast), 1)

# set state |00> to conditional probability of slow RV and small Diameter
qc.x(0)
qc.x(1)
ccry(qc,prob_to_angle(p_hazardous_slow_small),0,1,2)
qc.x(0)
qc.x(1)

# set state |01> to conditional probability of slow RV and large Diameter
qc.x(0)
ccry(qc,prob_to_angle(p_hazardous_slow_large),0,1,2)
qc.x(0)

# set state |10> to conditional probability of fast RV and small Diameter
qc.x(1)
ccry(qc,prob_to_angle(p_hazardous_fast_small),0,1,2)
qc.x(1)

# set state |11> to conditional probability of fast RV and large Diameter
ccry(qc,prob_to_angle(p_hazardous_fast_large),0,1,2)

qc.measure(qr[2], cr[0])
results = execute(qc,Aer.get_backend('qasm_simulator'), shots=1000).result().get_counts()
plot_histogram(results)

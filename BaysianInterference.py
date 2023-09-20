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

# Implementing Baysian Interference:

# Specifying the marginal probability
def prob_to_angle(prob):
    """
    Converts a given P(psi) value into an equivalent theta value.
    """
    return 2*asin(sqrt(prob))

# Baysian Interference

# Dataset with missing values
data = [
    (1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, None), (0, 1), (1, 0)
]

# The log‐likelihood function adapted for our data
def log_likelihood(data, prob_a_b, prob_a_nb, prob_na_b, prob_na_nb):
    def get_prob(point):    
        if point[0] == 1 and point[1] == 1:
            return log(prob_a_b)
        elif point[0] == 1 and point[1] == 0:
            return log(prob_a_nb)
        elif point[0] == 0 and point[1] == 1:
            return log(prob_na_b)
        elif point[0] == 0 and point[1] == 0:
            return log(prob_na_nb)
        else:
            return log(prob_na_b+prob_na_nb)

    return sum(map(get_prob, data))

# The as‐pqc function
def as_pqc(cnt_quantum, with_qc, cnt_classical=1, shots=1, hist=False, measure=False):
    # Prepare the circuit with qubits and a classical bit to hold the measurement
    qr = QuantumRegister(cnt_quantum)
    cr = ClassicalRegister(cnt_classical)
    qc = QuantumCircuit(qr, cr) if measure else QuantumCircuit(qr)

    with_qc(qc, qr=qr, cr=cr)
    
    results = execute(
        qc,
        Aer.get_backend('statevector_simulator') if measure is False else Aer.get_backend('qasm_simulator'),
        shots=shots
    ).result().get_counts()
    
    return plot_histogram(results, figsize=(12,4)) if hist else results

# The quantum Bayesian network
def qbn(data, hist=True): 
    def circuit(qc, qr=None, cr=None):
        list_a = list(filter(lambda item: item[0] == 1, data))
        list_na = list(filter(lambda item: item[0] == 0, data))
   
        # set the marginal probability of A
        qc.ry(prob_to_angle(
            len(list_a) / len(data)
        ), 0)

        # set the conditional probability of NOT A and (B / not B)
        qc.x(0)
        qc.cry(prob_to_angle(
            sum(list(map(lambda item: item[1], list_na))) /  len(list_na)
        ),0,1)
        qc.x(0)

        # set the conditional probability of A and (B / not B)
        qc.cry(prob_to_angle(
            sum(list(map(lambda item: item[1], list_a))) /  len(list_a)
        ),0,1)

    return as_pqc(2, circuit, hist=hist)

# Ignoring the missing data
qbn(list(filter(lambda item: item[1] is not None ,data)))

# Calculate the log‐likelihood when ignoring the missing data
def eval_qbn(model, prepare_data, data):
    results = model(prepare_data(data), hist=False)
    return (
        round(log_likelihood(data, 
            results['11'], # prob_a_b
            results['01'], # prob_a_nb
            results['10'], # prob_na_b
            results['00']  # prob_na_nb
        ), 3),
        results['10'] / (results['10'] + results['00'])
    )

print(eval_qbn(qbn, lambda dataset: list(filter(lambda item: item[1] is not None ,dataset)), data))

# Calculate the log‐likelihood when filling in 0
print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0) ,dataset)), data))

# Evaluating the guess
print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.5) ,dataset)), data))

# Refining the model
print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.3) ,dataset)), data))

# Further refining the model
print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.252) ,dataset)), data))

# Another iteration
print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.252) ,dataset)), data))

# positions of the qubits
QPOS_isLarge = 0
QPOS_fast = 1

def apply_islarge_fast(qc):
    # set the marginal probability of large Diameter
    qc.ry(prob_to_angle(p_large), QPOS_isLarge)

    # set the marginal probability of Fast Relative Velocity
    qc.ry(prob_to_angle(p_fast), QPOS_fast)


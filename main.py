import pandas as pd
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Loaded the Dataset
data = pd.read_csv  ('neo_v2.csv')

# Writing code for Pre-Procesing

# Lets drop unwated attributes such as Orbiting Body and Name
data = data.drop("orbiting_body", axis = 1)
data = data.drop("name", axis = 1)
data = data.drop("sentry_object", axis = 1)
data = data.drop("id", axis = 1)

# input_labels = data['hazardous'] 
# input_data = data.drop("hazardous", axis = 1)

# Splitted the Dataset into Training and Testing
train_input, test_input = train_test_split(data, test_size=0.3, random_state=0)

# Printing the Number of True and False Records in Train and Test Dataset
print("Train Dataset: No of True: {}, No. False: {}".format(len(train_input[train_input['hazardous'] == True]), len(train_input[train_input['hazardous'] == False])))
print("Test Dataset: No of True: {}, No. False: {}".format(len(test_input[test_input['hazardous'] == True]), len(test_input[test_input['hazardous'] == False])))

# Printing Number of records in training and testing
print("There are {} Training Records and {} Testing Records".format(train_input.shape[0], test_input.shape[0]))

# need to write preprocess function for calculating backward probability


## Processing -> on Quantum Circuit

# Create a quantum circuit with one qubit
qc = QuantumCircuit(1)

# Define initial_state as |1>
initial_state = [0,1]

# Apply initialization operation to the qubit at position 0
print(qc.initialize(initial_state, 0) )

# Tell Qiskit how to simulate our circuit
backend = Aer.get_backend('statevector_simulator') 

# Do the simulation, returning the result
result = execute(qc,backend).result()

'''
The execute function runs our quantum circuit (qc) at the specified backend. It returns a job object that has a useful method job.result(). This returns the result object once our program completes it.
'''

# get the probability distribution
counts = result.get_counts()

# Show the histogram
plot_histogram(counts)
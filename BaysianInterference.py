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

# print("\n\nThe Mean Max Diameter is: ", max_dia_mean)
# print("The Mean Relative Velocity is: ", Rv_mean)

for i,j, z in zip(data['est_diameter_max'], data['relative_velocity'], data['miss_distance']):
    if i>=max_dia_mean:
        cat_max_dia.append("Large")
    else :
        cat_max_dia.append("Small")
        
    if j>=Rv_mean:
        cat_RV.append("Fast") 
    else:
        cat_RV.append("Slow")

    if z>=0 and z< 25000000:
        cat_miss.append("Less") 
    elif z>= 25000000 and z<50000000:
        cat_miss.append("Medium")
    else:
        cat_miss.append("More")

processed_data = pd.DataFrame(list(zip(data['est_diameter_max'], data['relative_velocity'],data['miss_distance'], cat_max_dia, cat_RV,cat_miss, data['hazardous'])),columns=['Max_Diameter','Relative_Velocity','Miss_Distance', 'Categorized_Diameter', 'Categorized_Relative_Vel','Categorised_Miss_Distance','Hazardous'])

# Saving the Processed Data to get a better view
processed_data.to_csv('processedData.csv')

# Splitted the Dataset into Training and Testing
train_input, test_input = train_test_split(processed_data, test_size=0.3, random_state=0)

# # Printing Number of records in training and testing
# print("There are {} Training Records and {} Testing Records".format(train_input.shape[0], test_input.shape[0]))

# # Printing the Number of True and False Records in Train and Test Dataset
# print("Train Dataset: No of True: {}, No. False: {}".format(len(train_input[train_input['Hazardous'] == True]), len(train_input[train_input['Hazardous'] == False])))
# print("Test Dataset: No of True: {}, No. False: {}".format(len(test_input[test_input['Hazardous'] == True]), len(test_input[test_input['Hazardous'] == False])))

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

# print("\n\nPrinting Chances of Max Diameter Objects given they are hazardous:\n")
# print("{} Small Diameter objects had {} chances of being hazardous".format(pop_small, p_small))
# print("{} Large Diameter objects had {} chances of being hazardous".format(pop_large, p_large))

# For Relative Velocity:

# for Slow:
p_slow, pop_slow = prob_hazard_calc(train_input, "Categorized_Relative_Vel", "Slow")
# print(p_slow, pop_slow)

# for Fast:
p_fast, pop_fast = prob_hazard_calc(train_input, "Categorized_Relative_Vel", "Fast")
# print(p_fast, pop_fast)

# print("\n\nPrinting Chances of Relative Velocity Objects given they are hazardous:\n")
# print("{} Slow Relative Velocity objects had {} chances of being hazardous".format(pop_slow, p_slow))
# print("{} Fast Relative Velocity objects had {} chances of being hazardous".format(pop_fast, p_fast))

# For Miss Distance:

# For Less
p_less, pop_less = prob_hazard_calc(train_input, "Categorised_Miss_Distance", "Less")
# print(p_less, pop_less)

# For Medium
p_med, pop_med = prob_hazard_calc(train_input, "Categorised_Miss_Distance", "Medium")
# print(p_med, pop_med)

# For More:
p_more, pop_more = prob_hazard_calc(train_input, "Categorised_Miss_Distance", "More")
# print(p_more, pop_more)

# print("\n\nPrinting Chances of Miss Distance Objects given they are hazardous:\n")
# print("{} Less Miss Distance objects had {} chances of being hazardous".format(pop_less, p_less))
# print("{} Medium Miss Distance objects had {} chances of being hazardous".format(pop_med, p_med))
# print("{} More Miss Distance objects had {} chances of being hazardous".format(pop_more, p_more))

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

# print(eval_qbn(qbn, lambda dataset: list(filter(lambda item: item[1] is not None ,dataset)), data))

# # Calculate the log‐likelihood when filling in 0
# print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0) ,dataset)), data))

# # Evaluating the guess
# print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.5) ,dataset)), data))

# # Refining the model
# print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.3) ,dataset)), data))

# # Further refining the model
# print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.252) ,dataset)), data))

# # Another iteration
# print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.252) ,dataset)), data))

# positions of the qubits
QPOS_dia = 0
QPOS_RV = 1

def apply_islarge_fast(qc):
    # set the marginal probability of large Diameter
    qc.ry(prob_to_angle(p_large), QPOS_dia)

    # set the marginal probability of Fast Relative Velocity
    qc.ry(prob_to_angle(p_fast), QPOS_RV)

# Defining the CCRY‐gate:
def ccry(qc, theta, control1, control2, controlled):
    qc.cry(theta/2, control2, controlled)
    qc.cx(control1, control2)
    qc.cry(-theta/2, control2, controlled)
    qc.cx(control1, control2)
    qc.cry(theta/2, control1, controlled)

# Listing Represent the norm
# position of the qubit representing the norm
QPOS_NORM = 2

def apply_norm(qc, norm_params):
    """
    norm_params = {
        'p_norm_small_slow': 0.25,
        'p_norm_small_fast': 0.35,
        'p_norm_large_slow': 0.45,
        'p_norm_large_fast': 0.55
    }
    """

    # set the conditional probability of Norm given small/slow
    qc.x(QPOS_dia)
    qc.x(QPOS_RV)
    ccry(qc, prob_to_angle(
        norm_params['p_norm_small_slow']
    ),QPOS_dia, QPOS_RV, QPOS_NORM)
    qc.x(QPOS_dia)
    qc.x(QPOS_RV)

    # set the conditional probability of Norm given small/fast
    qc.x(QPOS_dia)
    ccry(qc, prob_to_angle(
        norm_params['p_norm_small_fast']
    ),QPOS_dia, QPOS_RV, QPOS_NORM)
    qc.x(QPOS_dia)

    # set the conditional probability of Norm given large/slow
    qc.x(QPOS_RV)
    ccry(qc, prob_to_angle(
        norm_params['p_norm_large_slow']
    ),QPOS_dia, QPOS_RV, QPOS_NORM)
    qc.x(QPOS_RV)

    # set the conditional probability of Norm given large/fast
    ccry(qc, prob_to_angle(
        norm_params['p_norm_large_fast']
    ),QPOS_dia, QPOS_RV, QPOS_NORM)

# Listing Calculate the probabilities related to the miss distance
pop_more = train_input[train_input.Categorised_Miss_Distance.eq("More")]
hazardous_more =  round(len(pop_more[pop_more.Hazardous.eq(1)])/len(pop_more), 2)
p_more = round(len(pop_more)/len(train_input), 2)

pop_med = train_input[train_input.Categorised_Miss_Distance.eq("Medium")]
hazardous_med =  round(len(pop_med[pop_med.Hazardous.eq(1)])/len(pop_med), 2)
p_med = round(len(pop_med)/len(train_input), 2)

pop_less = train_input[train_input.Categorised_Miss_Distance.eq("Less")]
hazardous_less =  round(len(pop_less[pop_less.Hazardous.eq(1)])/len(pop_less), 2)
p_less = round(len(pop_less)/len(train_input), 2)

# print("More Miss Distance: {} of the Objects, hazardous: {}".format(p_more , hazardous_more))
# print("Medium Miss Distance: {} of the Objects, hazardous: {}".format(p_med,hazardous_med))
# print("Less Miss Distance: {} of the Objects, hazardous: {}".format(p_less,hazardous_less))

# Listing Represent the miss-distance
# positions of the qubits
QPOS_more = 3
QPOS_med = 4
QPOS_less = 5

def apply_class(qc):
    # set the marginal probability of miss-distance=more
    qc.ry(prob_to_angle(p_more), QPOS_more)

    qc.x(QPOS_more)
    # set the marginal probability of Pclass=2nd
    qc.cry(prob_to_angle(p_med/(1-p_more)), QPOS_more, QPOS_med)

    # set the marginal probability of Pclass=3rd    
    qc.x(QPOS_med)
    ccry(qc, prob_to_angle(p_less/(1-p_more-p_med)), QPOS_more, QPOS_med, QPOS_less)
    qc.x(QPOS_med)
    qc.x(QPOS_more)

# Listing Represent hazardous
# position of the qubit
QPOS_hazardous = 6

def apply_hazardous(qc, hazardous_params):    
    """
    hazardous_params = {
        'p_hazardous_favoured_more': 0.3,
        'p_hazardous_favoured_med': 0.4,
        'p_hazardous_favoured_less': 0.5,
        'p_hazardous_unfavoured_more': 0.6,
        'p_hazardous_unfavoured_med': 0.7,
        'p_hazardous_unfavoured_less': 0.8
    }
    """

    # set the conditional probability of Survival given unfavored by norm
    qc.x(QPOS_NORM)
    ccry(qc, prob_to_angle(
        hazardous_params['p_hazardous_unfavoured_more']
    ),QPOS_NORM, QPOS_more, QPOS_hazardous)

    ccry(qc, prob_to_angle(
        hazardous_params['p_hazardous_unfavoured_med']
    ),QPOS_NORM, QPOS_med, QPOS_hazardous)

    ccry(qc, prob_to_angle(
        hazardous_params['p_hazardous_unfavoured_less']
    ),QPOS_NORM, QPOS_less, QPOS_hazardous)
    qc.x(QPOS_NORM)

    # set the conditional probability of hazardous given favored by norm
    ccry(qc, prob_to_angle(
        hazardous_params['p_hazardous_favoured_more']
    ),QPOS_NORM, QPOS_more, QPOS_hazardous)

    ccry(qc, prob_to_angle(
        hazardous_params['p_hazardous_favoured_med']
    ),QPOS_NORM, QPOS_med, QPOS_hazardous)

    ccry(qc, prob_to_angle(
        hazardous_params['p_hazardous_favoured_less']
    ),QPOS_NORM, QPOS_less, QPOS_hazardous)

# Listing The quantum bayesian network
QUBITS = 7

def qbn_neo(norm_params, hazardous_params, hist=True, measure=False, shots=1): 
    def circuit(qc, qr=None, cr=None):
        apply_islarge_fast(qc)
        apply_norm(qc, norm_params)
        apply_class(qc)
        apply_hazardous(qc, hazardous_params)

    return as_pqc(QUBITS, circuit, hist=hist, measure=measure, shots=shots)

# Listing Try the QBN
norm_params = {
    'p_norm_small_slow': 0.25,
    'p_norm_small_fast': 0.35,
    'p_norm_large_slow': 0.45,
    'p_norm_large_fast': 0.55
}

hazardous_params = {
    'p_hazardous_favoured_more': 0.3,
    'p_hazardous_favoured_med': 0.4,
    'p_hazardous_favoured_less': 0.5,
    'p_hazardous_unfavoured_more': 0.6,
    'p_hazardous_unfavoured_med': 0.7,
    'p_hazardous_unfavoured_less': 0.8
}

qbn_neo(norm_params, hazardous_params, hist=True)

# Listing Calculate the parameters of the norm
def calculate_norm_params(objects):
    # the different diameteric objects in our data
    pop_large = objects[objects.Categorized_Diameter.eq("Large")]
    pop_small = objects[objects.Categorized_Diameter.eq("Small")]

    # combinations of being a large object and Relative Velocity
    pop_small_slow = pop_small[pop_small.Categorized_Relative_Vel.eq('Slow')]
    pop_small_fast = pop_small[pop_small.Categorized_Relative_Vel.eq('Fast')]
    pop_large_slow = pop_large[pop_large.Categorized_Relative_Vel.eq('Slow')]
    pop_large_fast = pop_large[pop_large.Categorized_Relative_Vel.eq('Fast')]

    norm_params = {
        'p_norm_small_slow': pop_small_slow.Norm.sum() /  len(pop_small_slow),
        'p_norm_small_fast': pop_small_fast.Norm.sum() /  len(pop_small_fast),
        'p_norm_large_slow': pop_large_slow.Norm.sum() /  len(pop_large_slow),
        'p_norm_large_fast': pop_large_fast.Norm.sum() /  len(pop_large_fast),
    }

    return norm_params

# Listing Calculate the parameters of hazardous
def calculate_hazardous_params(objects):
    # all hazardous
    hazardous = objects[objects.Hazardous.eq(1)]
    
    # weight the object
    def weight_object(norm, missDistance):
        return lambda object: (object[0] if norm else 1-object[0]) * (1 if object[1] == missDistance else 0)

    # calculate the probability of being hazardous
    def calc_prob(norm, missDistance):
        return sum(list(map(
            weight_object(norm, missDistance),
            list(zip(hazardous['Norm'], hazardous['Categorised_Miss_Distance']))
        ))) / sum(list(map(
            weight_object(norm, missDistance), 
            list(zip(objects['Norm'], objects['Categorised_Miss_Distance']))
        )))
    
    hazardous_params = {
        'p_hazardous_favoured_more': calc_prob(True, "More"),
        'p_hazardous_favoured_med': calc_prob(True, "Medium"),
        'p_hazardous_favoured_less': calc_prob(True, "Less"),
        'p_hazardous_unfavoured_more': calc_prob(False, "More"),
        'p_hazardous_unfavoured_med': calc_prob(False, "Medium"),
        'p_hazardous_unfavoured_less': calc_prob(False, "Less")
    }

    return hazardous_params

# Listing Prepare the data
def prepare_data(objects, params):
    """
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
    """
    # is the object large?
    objects['IsLarge'] = objects['Categorized_Diameter'].map(lambda dia: 0 if dia == "Small" else 1)

    # the probability of favored by norm given diameter, relative velocity, and hazardous
    objects['Norm'] = list(map(
        lambda item: params['p_norm_{}_{}_{}'.format(
            'small' if item[0] == "Small" else 'large',
            'slow' if item[1] == "Slow" else 'fast',
            'nonhazardous' if item[2] == 0 else 'hazardous'
        )],
        list(zip(objects['IsLarge'], objects['Categorized_Relative_Vel'], objects['Hazardous']))
    ))
    return objects

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

# Listing Run the qbn
objects = prepare_data(train_input, params)
results = qbn_neo(calculate_norm_params(objects), calculate_hazardous_params(objects), hist=False)

# Listing Get a list of relevant states
def filter_states(states, position, value):
    return list(filter(lambda item: item[0][QUBITS-1-position] == str(value), states))

# Listing The states with hazardous objects
filter_states(results.items(), QPOS_hazardous, '1')

# Listing Calculate the marginal probability to be hazardous
def sum_states(states):
    return sum(map(lambda item: item[1], states))

sum_states(filter_states(results.items(), QPOS_hazardous, '1'))

# Listing The log‐likelihood function adapted for our data
def log_likelihood_neo(data, results):
    states = results.items()
    
    def calc_prob(norm_val, islarge_val, rv_val, hazardous_val):
        return sum_states(
            filter_states(
                filter_states(
                    filter_states(
                        filter_states(states, QPOS_RV, rv_val),
                        QPOS_dia, islarge_val
                    ), QPOS_hazardous, hazardous_val
                ), QPOS_NORM, norm_val))
        
    probs = {
        'p_favoured_large_slow_hazardous': calc_prob('1', '1', '0', '1'),
        'p_favoured_large_slow_nonhazardous': calc_prob('1', '1', '0', '0'),
        'p_favoured_large_fast_hazardous': calc_prob('1', '1', '1', '1'),
        'p_favoured_large_fast_nonhazardous': calc_prob('1', '1', '1', '0'),
        'p_favoured_small_slow_hazardous': calc_prob('1', '0', '0', '1'),
        'p_favoured_small_slow_nonhazardous': calc_prob('1', '0', '0', '0'),
        'p_favoured_small_fast_hazardous': calc_prob('1', '0', '1', '1'),
        'p_favoured_small_fast_nonhazardous': calc_prob('1', '0', '1', '0'),
        'p_unfavoured_large_slow_hazardous': calc_prob('0', '1', '0', '1'),
        'p_unfavoured_large_slow_nonhazardous': calc_prob('0', '1', '0', '0'),
        'p_unfavoured_large_fast_hazardous': calc_prob('0', '1', '1', '1'),
        'p_unfavoured_large_fast_nonhazardous': calc_prob('0', '1', '1', '0'),
        'p_unfavoured_small_slow_hazardous': calc_prob('0', '0', '0', '1'),
        'p_unfavoured_small_slow_nonhazardous': calc_prob('0', '0', '0', '0'),
        'p_unfavoured_small_fast_hazardous': calc_prob('0', '0', '1', '1'),
        'p_unfavoured_small_fast_nonhazardous': calc_prob('0', '0', '1', '0'),
    }

    return round(sum(map(
        lambda item: log(probs['p_{}_{}_{}_{}'.format(
                'unfavoured',
                'small' if item[1] == 0 else 'large',
                'slow' if item[2] == "Slow" else 'fast',
                'nonhazardous' if item[3] == 0 else 'hazardous'
            )] + probs['p_{}_{}_{}_{}'.format(
                'favoured',
                'small' if item[1] == 0 else 'large',
                'slow' if item[2] == "Slow" else 'fast',
                'nonhazardous' if item[3] == 0 else 'hazardous'
            )]
        ),
        list(zip(data['Norm'], data['IsLarge'], data['Categorized_Relative_Vel'], data['Hazardous']))
    )), 3)

# Listing Calculate the log‐likelihood
log_likelihood_neo(train_input, results)

# Listing Obtain new object values from the results
def to_params(results):
    states = results.items()
    
    def calc_norm(islarge_val, rv_val, hazardous_val):
        pop = filter_states(filter_states(filter_states(states, QPOS_RV, rv_val), QPOS_dia, islarge_val), QPOS_hazardous, hazardous_val)

        p_norm = sum(map(lambda item: item[1], filter_states(pop, QPOS_NORM, '1')))
        p_total = sum(map(lambda item: item[1], pop))
        return p_norm / p_total


    return {
        'p_norm_large_slow_hazardous': calc_norm('1', '0', '1'),
        'p_norm_large_slow_nonhazardous': calc_norm('1', '0', '0'),
        'p_norm_large_fast_hazardous': calc_norm('1', '1', '1'),
        'p_norm_large_fast_nonhazardous': calc_norm('1', '1', '0'),
        'p_norm_small_slow_hazardous': calc_norm('0', '0', '1'),
        'p_norm_small_slow_nonhazardous': calc_norm('0', '0', '0'),
        'p_norm_small_fast_hazardous': calc_norm('0', '1', '1'),
        'p_norm_small_fast_nonhazardous': calc_norm('0', '1', '0'),
    }

# Listing Calcualte new objects
to_params(results)

# Listing The recursive training automatism
def train_qbn_neo(objects, params, iterations):
    if iterations > 0:
        new_params = train_qbn_neo(objects, params, iterations - 1)

        objects = prepare_data(objects, new_params)
        results = qbn_neo(calculate_norm_params(objects), calculate_hazardous_params(objects), hist=False)

        # print ('The log-likelihood after {} iteration(s) is {}'.format(iterations, log_likelihood_neo(objects, results)))
        return to_params(results)
    
    return params

# Listing Train the QBN
trained_params = train_qbn_neo(train_input, {
    'p_norm_large_slow_hazardous': 0.45,
    'p_norm_large_slow_nonhazardous': 0.46,
    'p_norm_large_fast_hazardous': 0.47,
    'p_norm_large_fast_nonhazardous': 0.48,
    'p_norm_small_slow_hazardous': 0.49,
    'p_norm_small_slow_nonhazardous': 0.51,
    'p_norm_small_fast_hazardous': 0.52,
    'p_norm_small_fast_nonhazardous': 0.53,
}, 25)

# Listing The parameters after training
trained_params

# Listing Pre‐processing
def pre_process(object):
    return (object['IsLarge'] == 1, object['Categorized_Relative_Vel'] == 'Fast', object['Categorised_Miss_Distance'])

# Listing Apply the known data on the quantum circuit
def apply_known(qc, is_large, is_fast, missDistance):
    if is_large:
        qc.x(QPOS_dia)

    if is_fast:
        qc.x(QPOS_RV)
    
    qc.x(QPOS_more if missDistance == "More" else (QPOS_med if missDistance == "Medium" else QPOS_less))

# Listing Get the trained QBN
def get_trained_qbn(objects, params):

    prepared_objects = prepare_data(objects, params)
    norm_params = calculate_norm_params(prepared_objects)
    hazardous_params = calculate_hazardous_params(prepared_objects)

    def trained_qbn_neo(object):
        (is_large, is_fast, missDistance) = object

        def circuit(qc, qr, cr):
            apply_known(qc, is_large, is_fast, missDistance)
            apply_norm(qc, norm_params)
            apply_hazardous(qc, hazardous_params)
            
            qc.measure(qr[QPOS_hazardous], cr[0])
        
        return as_pqc(QUBITS, circuit, hist=False, measure=True, shots=100)

    return trained_qbn_neo

# Listing Post‐processing
def post_process(counts):
    """
    counts -- the result of the quantum circuit execution
    returns the prediction
    """
    #print (counts)
    p_hazardous = counts['1'] if '1' in counts.keys() else 0
    p_nonhazardous = counts['0'] if '0' in counts.keys() else 0

    #return int(list(map(lambda item: item[0], counts.items()))[0])
    return 1 if p_hazardous > p_nonhazardous else 0

# Preparing Report
def run(f_classify, x):
    return list(map(f_classify, x))

def specificity(matrix):
    return matrix[0][0]/(matrix[0][0]+matrix[0][1]) if (matrix[0][0]+matrix[0][1] > 0) else 0

def npv(matrix):
    return matrix[0][0]/(matrix[0][0]+matrix[1][0]) if (matrix[0][0]+matrix[1][0] > 0) else 0

def classifier_report(name, run, classify, input, labels):
    cr_predictions = run(classify, input)
    cr_cm = confusion_matrix(labels, cr_predictions)

    cr_precision = precision_score(labels, cr_predictions)
    cr_recall = recall_score(labels, cr_predictions)
    cr_specificity = specificity(cr_cm)
    cr_npv = npv(cr_cm)
    cr_level = 0.25*(cr_precision + cr_recall + cr_specificity + cr_npv)

    print('The precision score of the {} classifier is {:.2f}'
        .format(name, cr_precision))
    print('The recall score of the {} classifier is {:.2f}'
        .format(name, cr_recall))
    print('The specificity score of the {} classifier is {:.2f}'
        .format(name, cr_specificity))
    print('The npv score of the {} classifier is {:.2f}'
        .format(name, cr_npv))
    print('The information level is: {:.2f}'
        .format(cr_level))
    
# Listing Run the Quantum Naive Bayes Classifier
# redefine the run-function
def run(f_classify, data):
    return [f_classify(data.iloc[i]) for i in range(0,len(data))]

# get the simple qbn
trained_qbn = get_trained_qbn(train_input, trained_params)

# evaluate the Quantum Bayesian Network
# classifier_report("QBN",
#     run,
#     lambda object: post_process(trained_qbn(pre_process(object))),
#     objects,
#     train_input['Hazardous'])
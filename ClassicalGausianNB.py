# In this code we use Classical Computing Gaussian Naive Bayes Algorithm on Nasa Nearst Earth Object Data Set
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statistics as st
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import math
import seaborn as samandar

# Loaded the Dataset
data = pd.read_csv('neo_v2.csv')

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

# Calculating Probability of Hazardious true And false
p_hazard_true = len(train_input[train_input['hazardous'] == True])/len(train_input)
p_hazard_false = len(train_input[train_input['hazardous'] == False])/len(train_input)

# Printing the target class probility
print("P[True]: {:.3f}".format(p_hazard_true))
print("P[False]: {:.3f}".format(p_hazard_false))

# Calculating probabilities for Hazardous = True
row1 = []
# Calculating Mean and variance for est_diameter_min for hazardous = true
row1.append(st.mean(train_input[train_input['hazardous'] == True]['est_diameter_min']))
row1.append(st.variance(train_input[train_input['hazardous'] == True]['est_diameter_min']))
row1.append(st.mean(train_input[train_input['hazardous'] == True]['est_diameter_max']))
row1.append(st.variance(train_input[train_input['hazardous'] == True]['est_diameter_max']))
row1.append(st.mean(train_input[train_input['hazardous'] == True]['relative_velocity']))
row1.append(st.variance(train_input[train_input['hazardous'] == True]['relative_velocity']))
row1.append(st.mean(train_input[train_input['hazardous'] == True]['miss_distance']))
row1.append(st.variance(train_input[train_input['hazardous'] == True]['miss_distance']))
row1.append(st.mean(train_input[train_input['hazardous'] == True]['absolute_magnitude']))
row1.append(st.variance(train_input[train_input['hazardous'] == True]['absolute_magnitude']))

# Calculating Probabilities for Hazardous = False
row2 = []
# Calculating Mean and variance for est_diameter_min for hazardous = false
row2.append(st.mean(train_input[train_input['hazardous'] == False]['est_diameter_min']))
row2.append(st.variance(train_input[train_input['hazardous'] == False]['est_diameter_min']))
row2.append(st.mean(train_input[train_input['hazardous'] == False]['est_diameter_max']))
row2.append(st.variance(train_input[train_input['hazardous'] == False]['est_diameter_max']))
row2.append(st.mean(train_input[train_input['hazardous'] == False]['relative_velocity']))
row2.append(st.variance(train_input[train_input['hazardous'] == False]['relative_velocity']))
row2.append(st.mean(train_input[train_input['hazardous'] == False]['miss_distance']))
row2.append(st.variance(train_input[train_input['hazardous'] == False]['miss_distance']))
row2.append(st.mean(train_input[train_input['hazardous'] == False]['absolute_magnitude']))
row2.append(st.variance(train_input[train_input['hazardous'] == False]['absolute_magnitude']))

calculated_fields = pd.DataFrame([row1,row2], columns= ['Mean_est_diameter_min','Var_est_diameter_min', 'Mean_est_diameter_max','Var_est_diameter_max', 'Mean_relative_velocity','Var_relative_velocity', 'Mean_miss_distance', 'Var_miss_distance', 'Mean_absolute_magnitude', 'Var_absolute_magnitude'])

# Saving the Calculated fields to get a better view
calculated_fields.to_csv('calculated_fields.csv')

# Calculating Posterior Probability Function F(x) = (1/2πσ2)e(-(x-x̄)/2σ2)
def post_prob(mean, var, input_x):
    return ((1.0/math.sqrt(2.0*np.pi*var))*math.exp((-1.0*((input_x-mean)**2.0))/(2.0*var)))

# df.iloc[row, column]
# print(calculated_fields.iloc[0,1])

# Writing the Estimation function:
def cnb(input):
    # Post_T = (p_hazard_true*def()*)
    Post_T = p_hazard_true
    Post_F = p_hazard_false 
    
    # print("For Hazardous = True")
    # Calculating Posterior Probability for Hazardous = True
    for i in range(5):
        t1 = i*2
        t2 = t1+1
        val = post_prob(calculated_fields.iloc[0,t1],calculated_fields.iloc[0,t2], input[i])
        # print(val)
        # print("Post_T: {}".format(Post_T))
        Post_T *= val
    
    # print("For hazardous = False")
    # Calculating Posterior Probability for Hazardous = False
    for i in range(5):
        t1 = i*2
        t2 = t1+1
        val = post_prob(calculated_fields.iloc[1,t1],calculated_fields.iloc[1,t2], input[i])
        # print(val)
        # print("Post_F: {}".format(Post_F))
        Post_F *= val
    
    # print("Posterior Probability for Hazardous = True is: {}".format(Post_T))
    # print("Posterior Probability for Hazardous = False is: {}".format(Post_F))

    if Post_T > Post_F:
        return True
    else:
        return False
    
# Specificity
def specificity(matrix):
    return matrix[0][0]/(matrix[0][0]+ matrix[0][1]) if (matrix[0][0]+matrix[0][1]>0) else 0

# Negative Predictive Value(NPV)
def npv(matrix):
    return matrix[0][0]/(matrix[0][1]+ matrix[1][0]) if (matrix[0][1]+ matrix[1][0] > 0) else 0

# Logic for testing the algorithm
actual_outputs = test_input['hazardous']

predicted_outputs = []

for i in range(len(test_input)):
    predicted_outputs.append(cnb(test_input.iloc[i].values.flatten().tolist()))

# Creating Classifier Report function
def classifier_report():
    name = 'Classical Guassian Naive Bayes Algorithm'
    cr_prediction = predicted_outputs
    labels = actual_outputs 
    cr_cm = confusion_matrix(labels, cr_prediction)


    print("Confusion Matrix")
    print(cr_cm)

    cr_precision = precision_score(labels, cr_prediction)
    cr_recall = recall_score(labels, cr_prediction)
    cr_specificity = specificity(cr_cm)
    cr_npv = npv(cr_cm)
    cr_level = 0.25 * (cr_precision + cr_recall + cr_specificity + cr_npv)

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
    
classifier_report()

# Visualisation
correlation = train_input.corr()

# plot the heatmap
samandar.heatmap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns, annot=True)

# plot the clustermap
samandar.clustermap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns, annot=True)

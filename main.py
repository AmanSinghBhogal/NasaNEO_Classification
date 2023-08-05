import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Loaded the Dataset
data = pd.read_csv  ('neo_v2.csv')

# Lets drop unwated attributes such as Orbiting Body and Name
data = data.drop("orbiting_body", axis = 1)
data = data.drop("name", axis = 1)

# The Scaler returns the NumPy-array instead of a Pandas DataFrame

scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

input_data = data[:, 0:6]
input_labels = data[:, 7]

# Splitted the Dataset into Training and Testing
train_input, test_input, train_labels, test_labels = train_test_split(input_data, input_labels, test_size=0.3, random_state=0)

# Printing the Number of True and False Records in Train and Test Dataset
print("Train Dataset: No of True: {}, No. False: {}".format(train_labels.sum(), len(train_labels) -train_labels.sum()))
print("Test Dataset: No of True: {}, No. False: {}".format(test_labels.sum(), len(test_labels) - test_labels.sum()))

# Printing Number of records in training and testing
print("There are {} Training Records and {} Testing Records".format(train_input.shape[0], test_input.shape[0]))
print("There are {} input columns".format(train_input.shape[1]))
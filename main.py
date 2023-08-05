import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Loaded the Dataset
data = pd.read_csv  ('neo_v2.csv')

# Lets drop unwated attributes such as Orbiting Body and Name
data = data.drop("orbiting_body", axis = 1)
data = data.drop("name", axis = 1)

# Splitted the Dataset into Training and Testing
Train, Test = train_test_split(data, test_size=0.3, random_state=0)

'''
The Scaler returns the NumPy-array instead of a Pandas DataFrame
'''

scaler = MinMaxScaler()
scaler.fit(Train)
scaler.fit(Test)
Train = scaler.transform(Train)
Test = scaler.transform(Test)

# Creating train input data containing all attributes except the label attribute
train_input = Train[:, 0:6]
train_labels = Train[:, 7]

# Creating test input data containing all attributes except the label attribute
test_input = Test[:, 0:6]
test_labels = Test[:,7]

# Printing the Number of True and False Records in Train and Test Dataset
print("Train Dataset: No of True: {}, No. False: {}".format(train_labels.sum(), len(train_labels) -train_labels.sum()))
print("Test Dataset: No of True: {}, No. False: {}".format(test_labels.sum(), len(test_labels) - test_labels.sum()))
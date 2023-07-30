import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('neo_v2.csv')

Train, Test = train_test_split(data, test_size=0.3, random_state=0)

# Printing the Number of True and False Records in Train and Test Dataset
print("Train Dataset: No of True: {}, No. False: {}".format(Train['hazardous'].sum(), len(Train) -Train['hazardous'].sum()))
print("Test Dataset: No of True: {}, No. False: {}".format(Test['hazardous'].sum(), len(Test) -Test['hazardous'].sum()))
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Loaded the Dataset
data = pd.read_csv  ('neo_v2.csv')

# Lets drop unwated attributes such as Orbiting Body and Name
data = data.drop("orbiting_body", axis = 1)
data = data.drop("name", axis = 1)
data = data.drop("sentry_object", axis = 1)
data = data.drop("id", axis = 1)

# Splitted the Dataset into Training and Testing
train_input, test_input = train_test_split(data, test_size=0.3, random_state=1000)

y_train = train_input['hazardous']
train_input = train_input.drop("hazardous", axis = 1)

param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}
nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
nbModel_grid.fit(train_input, y_train)
print(nbModel_grid.best_estimator_)

y_test = test_input['hazardous']
test_input = test_input.drop("hazardous", axis = 1)

y_pred = nbModel_grid.predict(test_input)
print(y_pred)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred), ": is the accuracy score")
# from sklearn.metrics import precision_score
# print(precision_score(y_test, y_pred), ": is the precision score")
from sklearn.metrics import recall_score
print(recall_score(y_test, y_pred), ": is the recall score")
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred), ": is the f1 score")
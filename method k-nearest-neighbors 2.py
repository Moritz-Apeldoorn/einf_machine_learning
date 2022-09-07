import numpy as np
import pandas as pd
from statistics import mode
from collections import Counter
import sklearn.model_selection as modsel
column_names = ["sepal_lenght", "sepal-width", "petal-length", "petal-width", "class"]
data_set = pd.read_csv("iris.csv", names=column_names, header=None)
data_set = data_set.reindex(columns=["sepal_lenght", "sepal-width", "petal-length", "petal-width", "class"])
print(data_set.head())
print(data_set.tail())
print(data_set.describe())

matrix_x = data_set.iloc[0:data_set.shape[0], 0:data_set.shape[1] - 1].values
vector_y = data_set.iloc[0:data_set.shape[0], data_set.shape[1] - 1].values
x_train, x_test, y_train, y_test = modsel.train_test_split(matrix_x, vector_y, test_size=0.2, random_state=1, shuffle=True, stratify=vector_y)

input_data = np.array([1, 2, 3, 4])
neigh = list(map(lambda x: np.linalg.norm(input_data - x), x_train))
neighbors = []
for run in range(3):
    minimum = np.argmin(neigh)
    neighbors.append(y_train[minimum])
    del neigh[minimum]
prediction = mode(neighbors)
for element in Counter(neighbors).most_common(1):
    prediction = element[0]

error = 0
for index in range(len(x_test)):
    neigh = list(map(lambda x: np.linalg.norm(x_test[index] - x), x_test))
    neighbors = []
    for run in range(3):
        minimum = np.argmin(neigh)
        neighbors.append(y_test[minimum])
        del neigh[minimum]
    prediction = mode(neighbors)
    if prediction != y_test[index]:
        error += 1
accuracy = 1-error/len(y_test)

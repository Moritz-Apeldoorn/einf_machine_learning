import numpy as np
import pandas as pd
import sklearn.model_selection as modsel
column_names = ["sepal_lenght", "sepal-width", "petal-length", "petal-width", "class"]
data_set = pd.read_csv("iris.csv", names=column_names, header=None)
data_set = data_set.reindex(columns=["sepal_lenght", "sepal-width", "petal-length", "petal-width", "class"])
print(data_set.head())
print(data_set.tail())
print(data_set.describe())

labels = list(set(data_set.iloc[0:data_set.shape[0], -1].values))
centroids = []
matrix_x = data_set.iloc[0:data_set.shape[0], 0:data_set.shape[1] - 1].values
vector_y = data_set.iloc[0:data_set.shape[0], data_set.shape[1] - 1].values

x_train, x_test, y_train, y_test = modsel.train_test_split(matrix_x, vector_y, test_size=0.2, random_state=1, shuffle=True, stratify=vector_y)

print(x_train.shape,
      y_train.shape)

list_0 = np.concatenate((x_train, y_train[:, None]), axis=1)
for element in labels:
    centroids.append(sum(y[:-1] for y in list(filter(lambda x: x[-1] == element, list_0))) / len(list(filter(lambda x: x == element, y_train))))

error = 0
for index in range(len(x_test)):
    if labels[np.argmin(list(map(lambda x: np.linalg.norm(x_test[index] - x), centroids)))] != y_test[index]:
        error += 1
accuracy = 1-error/len(y_test)
print(accuracy)

predict = labels[np.argmin(list(map(lambda x: np.linalg.norm(np.array([1, 2, 3, 4]) - x), centroids)))]

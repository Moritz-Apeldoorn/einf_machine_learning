import numpy as np
import pandas as pd
from sklearn.neighbors import NearestCentroid
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

classifier = NearestCentroid()
classifier.fit(x_train, y_train)

accuracy = classifier.score(x_test, y_test)
print(accuracy)

prediction = classifier.predict(np.array([1, 2, 3, 4]).reshape(1, 4))[0]
print(prediction)

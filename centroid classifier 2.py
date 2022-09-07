import numpy
import pandas as pd
from sklearn.neighbors import NearestCentroid
import sklearn.model_selection as modsel
column_names = ["sepal_lenght", "sepal-width", "petal-length", "petal-width", "class"]
iris_data_set = pd.read_csv("iris.csv")
matrix_x = iris_data_set.iloc[0:150, 0:4].values
matrix_y = iris_data_set.iloc[0:150, 4].values
x_train, x_test, y_train, y_test = modsel.train_test_split(matrix_x, matrix_y, test_size=0.2, stratify=matrix_y)
clf = NearestCentroid()
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
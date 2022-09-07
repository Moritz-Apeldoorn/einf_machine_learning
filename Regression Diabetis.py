import pandas as pd
import numpy
import sklearn.linear_model as linmodel
import sklearn.model_selection as modsel

diabetes_data_set = pd.read_csv("diabetes_regression.csv", header=None)
print(diabetes_data_set.head())
print(diabetes_data_set.tail())
print(diabetes_data_set.describe())
print(diabetes_data_set.shape)

#bei count kann man die zeilen anzahl ansehen!
matrix_x_1 = diabetes_data_set.iloc[0:diabetes_data_set.shape[0], 0:5].values
matrix_x_2 = diabetes_data_set.iloc[0:diabetes_data_set.shape[0], 5:10].values
# axis=0 um es drunter anzuhängen und axis=1 um es daneben anzuhängen
matrix_x =numpy.concatenate((matrix_x_1, matrix_x_2), axis=1)
vector_y = diabetes_data_set.iloc[0:diabetes_data_set.shape[0], 10]
x_train, x_test, y_train, y_test = modsel.train_test_split(matrix_x, vector_y, test_size=0.2)

linear_regression = linmodel.Ridge()
linear_regression.fit(x_train, y_train)
print(linear_regression.coef_)
print(linear_regression.intercept_)
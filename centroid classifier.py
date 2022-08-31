import pandas, numpy, statistics
data_set = pandas.read_csv("iris.csv")
matrix_xy = data_set.iloc[0:150, 0:5].values
matrix_x = data_set.iloc[0:150, 0:4].values
vector_y = data_set.iloc[0:150, 4].values
input_data = [1, 2, 3, 4]
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
center = [numpy.array([0, 0, 0, 0]), numpy.array([0, 0, 0, 0]), numpy.array([0, 0, 0, 0])]
for element in matrix_xy:
    center[labels.index(element[-1])] += numpy.array(element[0:4])
print(center)

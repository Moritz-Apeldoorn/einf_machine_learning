import pandas, numpy
data_set = pandas.read_csv("iris.csv").values
input_data = numpy.array([1, 2, 3, 4])
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
center = [[numpy.array([0, 0, 0, 0]), 0], [numpy.array([0, 0, 0, 0]), 0], [numpy.array([0, 0, 0, 0]), 0]]
for element in data_set:
    index = labels.index(element[-1])
    center[index] = [center[index][0] + element[0:4], center[index][1] + 1]
print(labels[numpy.argmin(list(map(lambda x: numpy.linalg.norm(x[0]/x[1]-input_data), center)))])
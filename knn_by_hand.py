import pandas, numpy, statistics
data_set = pandas.read_csv("iris.csv")
matrix_x = data_set.iloc[0:150, 0:4].values
vector_y = data_set.iloc[0:150, 4].values
input_data = [1, 2, 3, 4]
neigh = list(map(lambda x: numpy.linalg.norm(input_data-x), matrix_x))
neighbors = []
for run in range(5):
    min = numpy.argmin(neigh)
    neighbors.append(vector_y[min])
    del neigh[min]
print(statistics.mode(neighbors))
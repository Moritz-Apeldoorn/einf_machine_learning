import numpy

matrix = numpy.zeros((6, 4))
vector = numpy.array([5, 0, -1, 2])
for row in range(6):
    for column in range(4):
        matrix[row][column] = row * 4 + column + 1
print(numpy.matmul(matrix, vector))
matrix_1 = numpy.matrix([[2, 1],
                        [6, 4]])
matrix_2 = numpy.matrix([[2, -0.5],
                        [-3, 1]])
print(numpy.matmul(matrix_1, matrix_2))
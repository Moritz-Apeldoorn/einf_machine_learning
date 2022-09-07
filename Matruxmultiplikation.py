import numpy
matrix_1 = numpy.matrix([[2, 1],
                        [6, 4]])
matrix_2 = numpy.matrix([[2, -0.5],
                        [-3, 1]])
# print(numpy.matmul(matrix_1, matrix_2))
print(matrix_1 @ matrix_2)
# print(matrix_1.T)
# print(numpy.linalg.inv(matrix_1))
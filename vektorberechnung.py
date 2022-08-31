import numpy

vector_1 = numpy.array([1, 2, 3, 4])
vector_2 = numpy.array([2, 3, 4, 5])
vector_1 = vector_1.reshape((4, 1))
vector_2 = vector_2.reshape((1, 4))
print(vector_1.ndim)
print(vector_1)
print(vector_1.T)
print(numpy.dot(vector_1, vector_2))
print(numpy.linalg.norm(vector_1))
# ==> nur linalg.norm
print(numpy.linalg.norm(vector_2))
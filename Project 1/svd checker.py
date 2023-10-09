# Singular-value decomposition
from numpy import array
from numpy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4]])
#print(A)
# SVD
U, s, VT = svd(A)
print('U = ')
print(U)
print('sigma = ')
print(s)
print('VT = ')
print(VT)
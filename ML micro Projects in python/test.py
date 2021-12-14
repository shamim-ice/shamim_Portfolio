import numpy as np
from matplotlib.pyplot import plot as plt


A = np.array([[1, 2], [3, 4], [5, 6]])
print(A)


M = np.mean(A.T, axis=1)
print(M)

C = A-M
print(C)

V = np.cov(C.T)
print(V)

val, vec = np.linalg.eig(V)
print(val, vec)

P = vec.T.dot(C.T)
print(P.T)



import numpy as np
from scipy.io import loadmat


data = loadmat("values2.mat")
main = data["values2"]     

sub = np.loadtxt("submatrix_1.csv", delimiter=",")


sub_matrix_index = None
for i in range(len(main)):
    if np.allclose(main[i], sub):
        sub_matrix_index = i
        break

target = 1.5 * sub
def manhattan_distance(target, main, skip_index=None):
    distances = []
    for i in range(main.shape[0]):
        if i == skip_index:
            distances.append(np.inf)
        else:
            distances.append(np.sum(np.abs(main[i] - target)))
    return np.array(distances)

d1 = manhattan_distance(target, main, sub_matrix_index)
idx1 = np.argmin(d1)

print("Manhattan Closest Index:", idx1)
print("Distance:", d1[idx1])
def minkowski_distance(target, main, p=3, skip_index=None):
    distances = []
    for i in range(main.shape[0]):
        if i == skip_index:
            distances.append(np.inf)
        else:
            distances.append(np.power(np.sum(np.abs(main[i] - target) ** p), 1/p))
    return np.array(distances)

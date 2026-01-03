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

def chebyshev_distance(target, main, skip_index=None):
    distances = []
    for i in range(main.shape[0]):
        if i == skip_index:
            distances.append(np.inf)
        else:
            distances.append(np.max(np.abs(main[i] - target)))
    return np.array(distances)
d5 = chebyshev_distance(target, main, skip_index=sub_matrix_index)
idx5 = np.argmin(d5)

print("Chebyshev Closest Index:", idx5)
print("Distance:", d5[idx5])
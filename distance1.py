import numpy as np
from scipy.io import loadmat
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = loadmat("values2.mat")
main = data["values2"]

sub = np.loadtxt("submatrix_1.csv", delimiter=",")

# Find submatrix index
sub_matrix_index = None
for i in range(len(main)):
    if np.allclose(main[i], sub):
        sub_matrix_index = i
        break

# Target
target = 1.5 * sub
target_vec = target.flatten().reshape(1, -1)

def cosine_distance_library(target_vec, main, skip_index=None):
    distances = []
    for i in range(main.shape[0]):
        if i == skip_index:
            distances.append(np.inf)
        else:
            m_vec = main[i].flatten().reshape(1, -1)
            cos_sim = cosine_similarity(target_vec, m_vec)[0][0]
            distances.append(1 - cos_sim)   
    return np.array(distances)

# Compute distance
d_lib = cosine_distance_library(target_vec, main, sub_matrix_index)
idx_lib = np.argmin(d_lib)

print("Cosine Closest Index (Library):", idx_lib)
print("Distance:", d_lib[idx_lib])
import numpy as np

# Transformation matrix
T = np.array([
    [0, -1, 0],
    [1,  1, 0],
    [0,  0, 1]
], dtype=int)

# Example reflection
hkl = np.array([2, 3, 4])

# Transform HKL
hkl_new = T @ hkl

print("Original HKL :", hkl)
print("New HKL      :", hkl_new)

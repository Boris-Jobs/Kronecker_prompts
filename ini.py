import numpy as np


def generate_matrices(n):

    target_matrix = np.ones((n, n))

    matrices = [np.zeros((n, n)) for _ in range(n)]

    for i in range(n):
        for j in range(n):
            matrices[(i + j) % n][i, j] = 1

    return matrices


n = 3
matrices = generate_matrices(n)
sum_matrix = np.sum(matrices, axis=0)

print("Generated Matrices:")
for idx, matrix in enumerate(matrices):
    print(f"Matrix {idx + 1}:\n{matrix}\n")

print("Sum of Matrices:")
print(sum_matrix)

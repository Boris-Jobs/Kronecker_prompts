import numpy as np

def generate_matrices(n):
    # 初始化全1矩阵
    target_matrix = np.ones((n, n))
    
    # 初始化一个全0矩阵的列表
    matrices = [np.zeros((n, n)) for _ in range(n)]
    
    # 分配1到每个矩阵的不同位置
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


# Загрузка нужных библиотек

import time
import matplotlib.pyplot as plt
from numba import cuda
import numpy as np


@cuda.jit  # декоратор для работы функии на GPU
def GPU_matMul(matrix_a, matrix_b, matrix_c):
    i, j = cuda.grid(2)
    if i < matrix_c.shape[0] and j < matrix_c.shape[1]:
        matrix_c[i, j] = 0
        for k in range(matrix_a.shape[1]):
            matrix_c[i, j] += matrix_a[i, k] * matrix_b[k, j]


def CPU_matMul(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    result_matrix = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))
    for i in range(result_matrix.shape[0]):
        for j in range(result_matrix.shape[1]):
            count = 0
            for k in range(result_matrix.shape[0]):
                count += matrix_a[i, k] * matrix_b[k, j]
            result_matrix[i, j] = count
    return result_matrix


if __name__ == '__main__':
    threads_per_block = (16, 16)
    blocks_per_grid = (512, 256)
    mas_size = np.linspace(100, 2000, 20, dtype=int)
    print(mas_size)
    array_time_CPU = []
    array_time_GPU = []
    for size in mas_size:
        random_matrix_A = np.random.randint(1, 6, size=(size, size))
        random_matrix_B = np.random.randint(1, 6, size=(size, size))
        random_matrix_C = np.zeros((size, size))
        print(f"A --> \n{random_matrix_A}")
        print(f"B --> \n{random_matrix_B}")

        start_CPU = time.time()
        result_CPU = CPU_matMul(random_matrix_A, random_matrix_B)
        end_CPU = time.time()
        execution_CPU = end_CPU - start_CPU
        print(
            f"Размерность --> {(size, size)}  \nРезультат: \n{result_CPU} \nВремя выполнения --> {execution_CPU} секунд")
        array_time_CPU.append(execution_CPU)
        start_GPU = time.time()

        A_gpu = cuda.to_device(random_matrix_A)
        B_gpu = cuda.to_device(random_matrix_B)
        C_gpu = cuda.to_device(random_matrix_C)

        print(threads_per_block, blocks_per_grid)

        GPU_matMul[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu)
        result_GPU = C_gpu.copy_to_host()

        end_GPU = time.time()
        execution_GPU = end_GPU - start_GPU
        array_time_GPU.append(execution_GPU)
        print(
            f"Размерность --> {(size, size)}  \nРезультат: \n{result_GPU} \nВремя выполнения --> {execution_GPU} секунд")
        print(f" Проверка равенства матриц {np.array_equal(result_GPU, result_CPU)}")

    plt.plot(mas_size, array_time_CPU, label='CPU', color='blue', linestyle='-', linewidth=2)
    plt.plot(mas_size, array_time_GPU, label='GPU', color='green', linestyle='-', linewidth=2)

    plt.title('Графики CPU и GPU')
    plt.xlabel('размерность')
    plt.ylabel('время')
    plt.legend()
    plt.show()

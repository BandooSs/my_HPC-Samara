import time
import matplotlib.pyplot as plt
from numba import cuda
import numpy as np


@cuda.jit  # декоратор для работы функии на GPU
def GPU_sumVectors(array_A, array_B, array_C):
    for i in range(cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x, array_A.shape[0],
                   cuda.blockDim.x * cuda.gridDim.x):
        if i < array_C.shape[0]:
            array_C[i] = array_A[i] + array_B[i]


def CPU_sumVectors(array_A: np.ndarray, array_B: np.ndarray) -> np.ndarray:
    array_C = np.zeros((array_A.shape[0],), dtype=float)
    for index in range(array_A.size):
        array_C[index] = array_A[index] + array_B[index]
    return array_C


def start_calculation(mas_size: np.ndarray, threads_per_block: int, blocks_per_grid: int):
    table_values_CPU = np.zeros((0, mas_size.shape[0]))
    table_values_GPU = np.zeros((0, mas_size.shape[0]))
    for i in range(0, N):
        array_time_CPU = []
        array_time_GPU = []
        for size in mas_size:
            start_CPU = time.time()
            array_A = np.random.normal(0.3, 1.5, size)
            array_B = np.random.normal(0.3, 1.5, size)

            result_CPU = CPU_sumVectors(array_A, array_B)
            end_CPU = time.time()
            execution_CPU = end_CPU - start_CPU
            print(
                f"Размерность --> {size}  \nРезультат на CPU: \n{result_CPU} \nВремя выполнения --> {execution_CPU} секунд\n")
            array_time_CPU.append(execution_CPU)

            start_GPU = time.time()

            A_gpu = cuda.to_device(array_A)
            B_gpu = cuda.to_device(array_B)
            C_gpu = cuda.to_device(np.zeros((size,), dtype=float))

            print(threads_per_block, blocks_per_grid)

            GPU_sumVectors[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu)
            result_GPU = C_gpu.copy_to_host()

            end_GPU = time.time()
            execution_GPU = end_GPU - start_GPU
            array_time_GPU.append(execution_GPU)
            print(
                f"Размерность --> {size}  \nРезультат на GPU: \n{result_GPU} \nВремя выполнения --> {execution_GPU} секунд")
            print(f"Проверка равенства матриц {np.array_equal(result_GPU, result_CPU)}\n")
        table_values_CPU = np.vstack((table_values_CPU, np.array(array_time_CPU).reshape((1, N))))
        table_values_GPU = np.vstack((table_values_GPU, np.array(array_time_GPU).reshape((1, N))))
    table_values_CPU = np.squeeze(table_values_CPU)
    table_values_GPU = np.squeeze(table_values_GPU)

    print(f"Таблица времени CPU : \n{table_values_CPU}\n")
    print(f"Таблица времени GPU : \n{table_values_GPU}\n")

    mas_CPU_time = [np.mean(table_values_CPU[:, i]) for i in range(table_values_CPU.shape[1])]
    mas_GPU_time = [np.mean(table_values_GPU[:, i]) for i in range(table_values_GPU.shape[1])]
    return mas_CPU_time, mas_GPU_time



if __name__ == '__main__':
    print("LR_1")
    threads_per_block = 512
    blocks_per_grid = 1024
    a = np.random.normal(0, 1.5, 1000)
    b = np.random.normal(0, 1.5, 1000)
    ag = cuda.to_device(a)
    ab = cuda.to_device(b)
    cg = cuda.to_device(np.zeros((1000,), dtype=float))

    print(threads_per_block, 3)

    GPU_sumVectors[3, threads_per_block](ag, ab, cg)
    result_GPU = cg.copy_to_host()

    N = 10
    mas_size = np.linspace(1000000, 10000000, N, dtype=int)

    mas_CPU_time, mas_GPU_time = start_calculation(mas_size, threads_per_block, blocks_per_grid)
    mas_CPU_time = np.array(mas_CPU_time)
    mas_GPU_time = np.array(mas_GPU_time)
    print(f"Время CPU : \n{mas_CPU_time}\n")
    print(f"Время GPU : \n{mas_GPU_time}\n")

    print(mas_CPU_time / mas_GPU_time)
    plt.ticklabel_format(axis='x', style='plain')
    plt.plot(mas_size, mas_CPU_time / mas_GPU_time, label='Ускорение', color='green', linestyle='-', linewidth=2)

    plt.title('Графики Ускорения')
    plt.xlabel('размерность')
    plt.ylabel('ускорение')
    plt.legend()
    plt.show()

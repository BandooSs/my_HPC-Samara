import math
import time

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numba import cuda


@cuda.jit
def gauss_GPU(x):
    return 0.2 * math.exp(-(x ** 2) / 10)

def gauss_CPU(x):
    return 0.2 * math.exp(-(x ** 2) / 10)

@cuda.jit  # декоратор для работы функии на GPU
def GPU_bilateral_filter(input_image, res):
    height, width = res.shape
    for i in range(cuda.grid(2)[0], height,
                   cuda.blockDim.x * cuda.gridDim.x):
        for j in range(cuda.grid(2)[1], width,
                       cuda.blockDim.y * cuda.gridDim.y):
            if i < height and j < width:
                current_pixel = input_image[i, j] / 255.0
                min_i = max(0, i - 4)
                max_i = min(height - 1, i + 4)
                min_j = max(0, j - 4)
                max_j = min(width - 1, j + 4)
                k = 0
                value = 0
                for x in range(min_i, max_i + 1):
                    for y in range(min_j, max_j + 1):
                        f = input_image[x, y] / 255.0
                        r = gauss_GPU(f - current_pixel)
                        dist_x = (x - i) ** 2
                        dist_y = (y - j) ** 2
                        g = gauss_GPU(dist_x + dist_y)
                        value += f * r * g
                        k += g * r
                res[i, j] = 255.0 * value / k


def CPU_bilateral_filter(input_image, kernel=4):
    height, width = input_image.size
    res = np.zeros((width, height), dtype=np.uint8)
    print(input_image.size)
    for i in range(height):
        for j in range(width):
            current_pixel = input_image.getpixel((i, j)) / 255.0
            min_i = max(0, i - kernel)
            max_i = min(height - 1, i + kernel)
            min_j = max(0, j - kernel)
            max_j = min(width - 1, j + kernel)
            k = 0
            value = 0
            for x in range(min_i, max_i + 1):
                for y in range(min_j, max_j + 1):
                    f = input_image.getpixel((x, y)) / 255.0
                    r = gauss_CPU(f - current_pixel)
                    dist_x = (x - i) ** 2
                    dist_y = (y - j) ** 2
                    g = gauss_CPU(dist_x + dist_y)
                    value += f * r * g
                    k += g * r
            res[j, i] = 255.0 * value / k
    return res


def start_calculation(w_size: np.ndarray, h_size: np.ndarray):
    table_values_CPU = np.zeros((0, w_size.shape[0]))
    table_values_GPU = np.zeros((0, w_size.shape[0]))
    image_path = "cat.bmp"
    img = Image.open(image_path).convert('L')
    for n in range(12):
        print(f"\nn = {n}")
        array_time_CPU = []
        array_time_GPU = []
        for w, h in zip(w_size, h_size):
            start_CPU = time.time()
            new_img = img.resize((w, h))
            result_CPU = CPU_bilateral_filter(new_img)
            end_CPU = time.time()
            execution_CPU = end_CPU - start_CPU
            print(
                f"Размерность --> {(w, h)}  \nРезультат на CPU: \n{result_CPU} \nВремя выполнения --> {execution_CPU} секунд\n")
            PIL_image = Image.fromarray(result_CPU, mode='L')
            PIL_image.save(f'CPU_{str(w)}x{str(h)}.bmp')
            array_time_CPU.append(execution_CPU)

            start_GPU = time.time()
            img_array = np.array(new_img)
            GPU_img_array = cuda.to_device(img_array)
            result_GPU = np.zeros((h, w), dtype=np.uint8)
            result_GPU = cuda.to_device(result_GPU)

            GPU_bilateral_filter[blocks_per_grid, threads_per_block](GPU_img_array, result_GPU)
            result_GPU = result_GPU.copy_to_host()
            end_GPU = time.time()
            execution_GPU = end_GPU - start_GPU
            array_time_GPU.append(execution_GPU)

            print(
                f"Размерность --> {(w, h)}  \nРезультат на GPU: \n{result_GPU} \nВремя выполнения --> {execution_GPU} секунд")
            print(f"Проверка равенства матриц {np.array_equal(result_GPU, result_CPU)}\n")

            PIL_image = Image.fromarray(result_GPU, mode='L')
            PIL_image.save(f'GPU_{str(w)}x{str(h)}.bmp')
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
    print("LR_2")
    threads_per_block = (4, 4)
    blocks_per_grid = (8, 8)
    image_path = "cat.bmp"
    img = Image.open(image_path).convert('L')
    print(img.size)
    N = 10
    w_size = np.linspace(100, img.size[0], N, dtype=int)
    h_size = np.linspace(100, img.size[1], N, dtype=int)
    mas_size = [str((i, j)) for i, j in zip(w_size, h_size)]

    mas_CPU_time, mas_GPU_time = start_calculation(w_size, h_size)

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
    plt.xticks(np.linspace(0, 10, 10), mas_size)
    plt.legend()
    plt.show()

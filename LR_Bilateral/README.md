
# Лабораторная работая  "Bilateral"
## Задача: 
- Реализовать билатеральный фильтр Язык: C++ или Python.<br />
    - Входные данные: изображение с различными разрешением.<br />
    - Выходные данные: проверка корректности  + время вычисления.
    - Характеристика системы: 
        - 1. видеокарта: GTX 1050;
        - 2. процессор: AMD Ryzen 5 2600 Six-Core Processor 3.40 GHz. 

В данной лабораторной работе производился запуск программы для сложения двух векторов с использованием технологии CUDA. В программе необходимо применить билатеральный фильтрна изображениях с различными расзрешениями. Измерения проводились на размерностях GridDim (2,2) и BlockDim(5,5)( всего получается 100 нитей) и  GridDim (4,4) и BlockDim(8,8) (всего получается 1024 нити). Код программы представлен в файле. Использовалась библиотека numba.  В таблицах будут представлены усредненые значения по 12 запускам.<br />
Задачи распараллеливание на CUDA:
1. считывание фото на CPU;
2. переброска матрицы на GPU;
3. расчет на GPU;
4. переброска результата на CPU.   

В таблице представлены результаты -->
| Размерность | Время работы CPU в секундах | Время работы GPU в секундах <br /> GridDim (2,2),BlockDim(5,5) | Время работы GPU в секундах <br /> GridDim (4,4),BlockDim(8,8) |
|-------------|:----------------:|-----------------:|-----------------:|
| 100x100   | 1.10600273     | 0.28119548 (3.93321657)  | 0.2309804  (4.78829696)   |
| 302x231   | 8.05266444     | 0.32629665 (24.67896775) | 0.03503211 (229.86521965) |
| 504x362   | 21.44752566    | 0.38635119 (55.51303147) | 0.0857443  (250.13353996) |
| 706x493   | 40.841187      | 0.94419289 (43.25513101) | 0.16248051 (251.3605243)  |
| 908x624   | 67.20372613    | 1.25580104 (53.51462842) | 0.24755915 (271.46532905) |
| 1111x755  | 101.05069097	 | 1.77756834 (56.84771082) | 0.35917401 (281.34187684) |
| 1313x886  | 138.91861677	 | 2.51820477 (55.16573494) | 0.50059573 (277.50659633) |
| 1515x1017 | 187.08229502	 | 3.26177382 (57.35599863) | 0.65736437 (284.59451714) |
| 1717x1148 | 239.17136494	 | 4.06737598 (58.80237437) | 0.83597962 (286.09712371) |
| 1920x1280 | 296.15654937	 | 4.98646331 (59.3921044)  | 1.05573932 (280.52052513) |





На рисунке предствален график ускорения работы  программы на GPU по сравнению с CPU  при GridDim (2,2) и BlockDim(5,5): 

![График](https://github.com/BandooSs/my_HPC-Samara/blob/main/LR_Bilateral/2x2.jpg)

На рисунке предствален график ускорения работы  программы на GPU по сравнению с CPU  при GridDim (4,4) и BlockDim(8,8): 

![График](https://github.com/BandooSs/my_HPC-Samara/blob/main/LR_Bilateral/4x4.jpg)



Результаты указывают на то, что при увеличении размерности GridDim и BlockDim  ускорение работы на GPU становится более большим, при этом можно сказать, что идельаный случай для расчета на GPU, когда число поток будет равно размерности матрицы. Это позволило бы одному потоку отработать одну операцию и закончить свою работу, нежели ему придется считать несколько операций. Максимального ускорение (286.09712371) удалось достичь при  GridDim (4,4),BlockDim(8,8) при размерности 1717x1148. 
Пример работы представлен ниже CPU: <br />
![Рисунок](https://github.com/BandooSs/my_HPC-Samara/blob/main/LR_Bilateral/CPU_100x100.bmp)


Пример работы представлен ниже GPU: <br />
![Рисунок](https://github.com/BandooSs/my_HPC-Samara/blob/main/LR_Bilateral/GPU_100x100.bmp)
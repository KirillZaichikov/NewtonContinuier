'''Надо сделать отдельный файл для запуска, не дописал'''

from utils.integrator import *
import numpy as np

dim = 3  # Размерность

# Начальные условия
x_start = np.array([3.6081561904, 0.7210613443, -0.4547532469])  # Неподвижная точка при 830 0.485

# Начальные параметры
alpha0 = 0.485
beta0 = 773

# Диапазоны параметров
# alpha_min, alpha_max = 0.41, 0.485
# beta_min, beta_max = 735, 760
# alpha_min, alpha_max = 0.35, 0.55
# beta_min, beta_max = 660, 830
alpha_min, alpha_max = 0.4, 0.5
beta_min, beta_max = 760, 790

# Шаги
# alpha_step = 0.0004
# beta_step = 0.32
alpha_step = 0.0012
beta_step = 0.96

print("Инициализация системы...")
print(f"Размерность: {dim}")
print(f"Начальная точка: {x_start}")
print(f"Начальные параметры: alpha={alpha0}, beta={beta0}")
print(f"Результатом будет сетка {(alpha_max-alpha_min)/alpha_step}x{(beta_max-beta_min)/beta_step}")

# Протягиватель
continuation = FixedPointContinuation(test_F, dim=dim)

# Выполняем полное протягивание
results = continuation.full_continuation(
    x_start=x_start,
    alpha0=alpha0, beta0=beta0,
    alpha_min=alpha_min, alpha_max=alpha_max, alpha_step=alpha_step,
    beta_min=beta_min, beta_max=beta_max, beta_step=beta_step,
    n_workers=8
)

# Сохраним полученную сетку в файл (сетка уже отсортирована)
with open("pointsGon.txt", "w", encoding="utf-8") as f:
    for item in results['all_points_MG']:
        alpha = item['alpha']
        print(alpha)
        beta = item['beta']
        coords = item['x']

        if len(coords)<6:
            print("Wrong data", alpha, beta, coords)
        
        line_values = [alpha, beta] + coords.tolist()
        line = ", ".join(map(str, line_values))

        f.write(line + "\\n")

# Визуализируем результаты
visualize_results(results)
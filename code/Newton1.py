# from utils.integrator import *

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import time


# Игнорируем предупреждения о делении на ноль в методе Ньютона
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

class NewtonSolver:
    """Класс для решения нелинейных уравнений методом Ньютона"""
    
    def __init__(self, dim: int = 5, tol: float = 0.0001, max_iter: int = 60, eps: float = 1e-8):
        """
        Инициализация NewtonSolver
        
        Parameters:
        -----------
        dim : int
            Размерность пространства
        tol : float
            Допуск для сходимости
        max_iter : int
            Максимальное число итераций
        eps : float
            Малое число для вычисления конечно-разностного якобиана
        """
        self.dim = dim
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps
    
    def finite_difference_jacobian(self, G: Callable, x: np.ndarray, 
                                   param1: float, param2: float) -> np.ndarray:
        """
        Вычисление якобиана методом конечных разностей
        
        Parameters:
        -----------
        G : Callable
            Функция G(x, param1, param2) = F(x, param1, param2) - x
        x : np.ndarray
            Точка, в которой вычисляется якобиан
        param1 : float
            Значение параметра param1
        param2 : float
            Значение параметра param2
            
        Returns:
        --------
        J : np.ndarray
            Матрица Якоби размерности (dim, dim)
        """
        J = np.zeros((self.dim, self.dim))
        G0 = G(x, param1, param2)
        
        for j in range(self.dim):
            dx = np.zeros(self.dim)
            dx[j] = self.eps

            G_perturbed = G(x+dx, param1, param2)
            J[:, j] = (G_perturbed - G0) / self.eps
            
        return J
    
    def solve(self, G: Callable, x0: np.ndarray, 
              param1: float, param2: float) -> Tuple[Optional[np.ndarray], int, bool]:
        """
        Решение уравнения G(x, param1, param2) = 0 методом Ньютона
        
        Parameters:
        -----------
        G : Callable
            Функция G(x, param1, param2) = F(x, param1, param2) - x
        x0 : np.ndarray
            Начальное приближение
        param1 : float
            Значение параметра param1
        param2 : float
            Значение параметра param2
            
        Returns:
        --------
        x_sol : Optional[np.ndarray]
            Найденное решение или None, если не сошлось
        iterations : int
            Число выполненных итераций
        success : bool
            Флаг успешности сходимости
        """
        iteration = 0
        if x0 is None:
            print(f"Итерации остановлены {os.getpid()} x0 - None:", param1, param2)
            return None, iteration, False
        x = x0.copy()
        
        for iteration in range(self.max_iter):
            # Вычисляем G(x)
            try:
                G_val = G(x, param1, param2)
            except RuntimeError as e:
                print(f"Итерации остановлены {os.getpid()}:", e)
                return None, iteration, False
            
            # Проверяем сходимость
            if np.linalg.norm(G_val) < self.tol:
                return x, iteration + 1, True
            
            # Вычисляем якобиан
            try:
                J = self.finite_difference_jacobian(G, x, param1, param2)
            except RuntimeError as e:
                print(f"Итерации остановлены {os.getpid()}:", e)
                return None, iteration, False
            
            # Решаем линейную систему J * dx = -G(x)
            try:
                dx = np.linalg.solve(J, -G_val)
            except np.linalg.LinAlgError:
                # Если матрица вырожденная, пытаемся использовать псевдообратную
                try:
                    dx = -np.linalg.pinv(J) @ G_val
                except:
                    print(f"Итерации остановлены {os.getpid()}:", "линейная система не решается")
                    return None, iteration + 1, False
            
            # Обновляем решение
            x += dx

            # Проверяем на расходимость (очень большие значения)
            if np.any(np.abs(x) > 1e10):
                print(f"Итерации остановлены {os.getpid()}:", "Ушла на бесконечность")
                return None, iteration + 1, False
        
        # Если вышли за пределы максимального числа итераций
        try:
            G_val = G(x, param1, param2)
        except RuntimeError as e:
            print(f"Итерации остановлены {os.getpid()}:", e)
            return None, iteration, False
        
        if np.linalg.norm(G_val) < self.tol:
            return x, self.max_iter, True
        else:
            print(f"Итерации остановлены {os.getpid()}:", "Максимальное число итераций")
            return None, self.max_iter, False

def process_point_worker(args):
    """
    Функция выполняющаяся параллельно

    Parameters:
    -----------
    args : list
        Параметры протягивания [obj, point, beta_min, beta_max, beta_step]

    Returns:
    --------
    x_sol : list
        Протянутые точки вдоль параметра beta
    """
    obj, point, beta_min, beta_max, beta_step = args

    # print("PID:", os.getpid())

    alpha = point['alpha']
    x_alpha = point['x']
    beta0 = point['beta']
    start = time.time()
    down_branch, up_branch = obj.vertical_continuation_for_alpha(
        x_alpha, alpha, beta0, beta_min, beta_max, beta_step
    )
    t = time.time()-start

    full_line = []

    if down_branch:
        down_branch_sorted = sorted(down_branch, key=lambda p: p['beta'])
        full_line.extend(down_branch_sorted)

    if up_branch:
        up_branch_sorted = sorted(up_branch, key=lambda p: p['beta'])
        full_line.extend(up_branch_sorted)

    # print(f"Поток {os.getpid()} закончил вычисление")
    # print(f"Поток {os.getpid()} количество точек {len(full_line)}")
    # print(f"Поток {os.getpid()} время на протягивание {t}")

    return full_line

class FixedPointContinuation:
    """Класс для протягивания неподвижных точек по параметрам"""
    
    def __init__(self, F: Callable, dim: int = 3):
        """
        Инициализация FixedPointContinuation
        
        Parameters:
        -----------
        F : Callable
            Функция F(x, alpha, beta)
        dim : int
            Размерность пространства
        """
        self.F = F
        self.dim = dim
        self.solver = NewtonSolver(dim=dim)
        
    def G(self, x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        Функция G(x, alpha, beta) = F(x, alpha, beta) - x
        
        Parameters:
        -----------
        x : np.ndarray
            Вектор состояния
        alpha : float
            Первый параметр
        beta : float
            Второй параметр
            
        Returns:
        --------
        np.ndarray
            Значение функции G
        """
        return self.F(x, alpha, beta) - x
    
    def horizontal_continuation(self, x_start: np.ndarray, 
                               alpha0: float, beta0: float,
                               alpha_min: float, alpha_max: float,
                               alpha_step: float) -> Tuple[List[dict], List[dict]]:
        """
        Горизонтальное протягивание вдоль alpha при фиксированном beta
        
        Parameters:
        -----------
        x_start : np.ndarray
            Начальное приближение при (alpha0, beta0)
        alpha0 : float
            Начальное значение alpha
        beta0 : float
            Фиксированное значение beta
        alpha_min : float
            Минимальное значение alpha
        alpha_max : float
            Максимальное значение alpha
        alpha_step : float
            Шаг по alpha
            
        Returns:
        --------
        left_branch : List[dict]
            Ветвь влево (alpha уменьшается)
        right_branch : List[dict]
            Ветвь вправо (alpha увеличивается)
        """
        # Ветвь влево (alpha уменьшается)
        left_branch = []
        current_x = x_start.copy()
        current_alpha = alpha0
        
        # Сохраняем начальную точку
        sol, iterations, success = self.solver.solve(self.G, current_x, alpha0, beta0)
        if success:
            current_x = sol
            left_branch.append({
                'alpha': alpha0,
                'beta': beta0,
                'x': sol.copy(),
                'iterations': iterations,
                'success': success
            })

        

        # Протягиваем влево
        alpha = alpha0 - alpha_step
        success = True
        while alpha >= alpha_min - 1e-10:
            # Используем решение с предыдущего шага как начальное приближение
            print(f"Поиск точки {alpha}")
            if success:
                sol, iterations, success = self.solver.solve(self.G, current_x, alpha, beta0)
            if success:
                current_x = sol
                left_branch.append({
                    'alpha': alpha,
                    'beta': beta0,
                    'x': sol.copy(),
                    'iterations': iterations,
                    'success': success
                })
                alpha -= alpha_step
            else:
                # Заполняю пустоту nan
                left_branch.append({
                    'alpha': alpha,
                    'beta': beta0,
                    'x': np.array([np.nan]*self.dim),
                    'iterations': iterations,
                    'success': success
                })
                alpha -= alpha_step
        
        # Ветвь вправо (alpha увеличивается)
        right_branch = []
        current_x = x_start.copy()
        current_alpha = alpha0
        
        # Протягиваем вправо
        alpha = alpha0 + alpha_step
        success = True
        while alpha <= alpha_max + 1e-10:
            print(f"Поиск точки {alpha}")
            if success:
                sol, iterations, success = self.solver.solve(self.G, current_x, alpha, beta0)
            if success:
                current_x = sol
                right_branch.append({
                    'alpha': alpha,
                    'beta': beta0,
                    'x': sol.copy(),
                    'iterations': iterations,
                    'success': success
                })
                alpha += alpha_step
            else:
                # Заполняю пустоту nan
                right_branch.append({
                    'alpha': alpha,
                    'beta': beta0,
                    'x': np.array([np.nan]*self.dim),
                    'iterations': iterations,
                    'success': success
                })
                alpha += alpha_step
        
        return left_branch, right_branch
    
    def vertical_continuation_for_alpha(self, x_alpha: np.ndarray, 
                                       alpha: float, beta0: float,
                                       beta_min: float, beta_max: float,
                                       beta_step: float) -> Tuple[List[dict], List[dict]]:
        """
        Вертикальное протягивание вдоль beta для фиксированного alpha
        
        Parameters:
        -----------
        x_alpha : np.ndarray
            Начальное приближение при (alpha, beta0)
        alpha : float
            Фиксированное значение alpha
        beta0 : float
            Начальное значение beta
        beta_min : float
            Минимальное значение beta
        beta_max : float
            Максимальное значение beta
        beta_step : float
            Шаг по beta
            
        Returns:
        --------
        down_branch : List[dict]
            Ветвь вниз (beta уменьшается)
        up_branch : List[dict]
            Ветвь вверх (beta увеличивается)
        """
        # Ветвь вниз (beta уменьшается)
        down_branch = []
        current_x = x_alpha.copy()
        
        # Проверяем начальную точку
        sol, iterations, success = self.solver.solve(self.G, current_x, alpha, beta0)
        if success:
            current_x = sol
            down_branch.append({
                'alpha': alpha,
                'beta': beta0,
                'x': sol.copy(),
                'iterations': iterations,
                'success': success
            })
        else:
            current_x = sol
            down_branch.append({
                'alpha': alpha,
                'beta': beta0,
                'x': np.array([np.nan]*self.dim),
                'iterations': iterations,
                'success': success
            })
        
        # Протягиваем вниз
        beta = beta0 - beta_step
        while beta >= beta_min - 1e-10:
            if success:
                sol, iterations, success = self.solver.solve(self.G, current_x, alpha, beta)    
            if success:
                current_x = sol
                down_branch.append({
                    'alpha': alpha,
                    'beta': beta,
                    'x': sol.copy(),
                    'iterations': iterations,
                    'success': success
                })
                beta -= beta_step
            else:
                # Заполняю пустоту nan
                down_branch.append({
                    'alpha': alpha,
                    'beta': beta,
                    'x': np.array([np.nan]*self.dim),
                    'iterations': iterations,
                    'success': success
                })
                beta -= beta_step
        
        # Ветвь вверх (beta увеличивается)
        up_branch = []
        current_x = x_alpha.copy()
        
        # Протягиваем вверх
        beta = beta0 + beta_step
        success = True # Флаг мог остаться False от протягивания вниз
        while beta <= beta_max + 1e-10:
            if success:
                sol, iterations, success = self.solver.solve(self.G, current_x, alpha, beta)
            if success:
                current_x = sol
                up_branch.append({
                    'alpha': alpha,
                    'beta': beta,
                    'x': sol.copy(),
                    'iterations': iterations,
                    'success': success
                })
                beta += beta_step
            else:
                # Заполняю пустоту nan
                up_branch.append({
                    'alpha': alpha,
                    'beta': beta,
                    'x': np.array([np.nan]*self.dim),
                    'iterations': iterations,
                    'success': success
                })
                beta += beta_step
        
        return down_branch, up_branch
    
    def parallel_vertical_continuation(self, horizontal_points, beta_min, beta_max, beta_step, n_workers=10):
        args = [
            (self, point, beta_min, beta_max, beta_step)
            for point in horizontal_points
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            vertical_lines = list(executor.map(process_point_worker, args))
            vl = []
            for elements in vertical_lines:
                vl.append(elements)

        return vl

    def full_continuation(self, x_start: np.ndarray, 
                         alpha0: float, beta0: float,
                         alpha_min: float, alpha_max: float, alpha_step: float,
                         beta_min: float, beta_max: float, beta_step: float,
                         n_workers: int = 8) -> dict:
        """
        Полное протягивание по двум параметрам
        
        Parameters:
        -----------
        x_start : np.ndarray
            Начальное приближение
        alpha0, beta0 : float
            Начальные значения параметров
        alpha_min, alpha_max, alpha_step : float
            Диапазон и шаг по alpha
        beta_min, beta_max, beta_step : float
            Диапазон и шаг по beta
        n_workers : int
            Число потоков для параллельных вычислений
            
        Returns:
        --------
        result : dict
            Словарь с результатами
        """
        print("Фаза 1: Горизонтальное протягивание...")
        start = time.time()
        left_branch, right_branch = self.horizontal_continuation(
            x_start, alpha0, beta0, alpha_min, alpha_max, alpha_step
        )
        print("Время на горизонтальное протягивание", time.time()-start)
        
        # Объединяем горизонтальную линию
        start = time.time()
        horizontal_line = []
        if left_branch:
            # Сортируем по возрастанию alpha
            left_branch_sorted = sorted(left_branch, key=lambda p: p['alpha'])
            horizontal_line.extend(left_branch_sorted)
        
        # Добавляем правую ветвь
        if right_branch:
            horizontal_line.extend(right_branch)
        print("Время на подготовку горизонтального списка", time.time()-start)

        print(f"Найдено {len(horizontal_line)} точек на горизонтальной линии")
        
        print("Фаза 2: Вертикальное протягивание...")
        start = time.time()
        vertical_lines = self.parallel_vertical_continuation(
            horizontal_line, beta_min, beta_max, beta_step, n_workers
        )
        print("Время на вертикальное протягивание", time.time()-start)

        # Собираем все точки в сетку
        all_points = []
        for line in vertical_lines:
            all_points.extend(line)
        print(f"Всего найдено {len(all_points)} точек c неподвижной точкой")
        return {
            'horizontal_line': horizontal_line,
            'vertical_lines': vertical_lines,
            'all_points': all_points
        }

def test_F(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Отображение

    Parameters:
    -----------
    x : np.ndarray
        Начальное условие
    alpha : float
        Значение параметра alpha
    beta : float
        Значение параметра beta
        
    Returns:
    --------
    np.array([x_, y_, z_])
        Следующая итерация
    """
    B = 0.7
    M_1, M_2 = alpha, beta
    x0, y0, z0 = x
    x_ = y0
    y_ = z0
    z_ = M_1 + B * x0 + M_2 * y0 - z0 ** 2

    return np.array([x_, y_, z_])

def visualize_results(results: dict):
    """
    Визуализация результатов протягивания
    
    Parameters:
    -----------
    results : dict
        Результаты, возвращенные методом full_continuation
    """
    all_points = results['all_points']
    
    if not all_points:
        print("Нет точек для визуализации")
        return
    
    alphas = []
    betas = []
    iterations = []
    success_flags = []
    
    for point in all_points:
        alphas.append(point['alpha'])
        betas.append(point['beta'])
        iterations.append(point['iterations'])
        success_flags.append(1 if point['success'] else 0)
    
    alphas = np.array(alphas)
    betas = np.array(betas)
    iterations = np.array(iterations)
    success_flags = np.array(success_flags)
    
    if len(alphas) > 1 and len(betas) > 1:
        alpha_unique = np.unique(alphas)
        beta_unique = np.unique(betas)
        
        if len(alpha_unique) > 1 and len(beta_unique) > 1:
            alpha_grid, beta_grid = np.meshgrid(alpha_unique, beta_unique)
            iterations_grid = np.full_like(alpha_grid, np.nan, dtype=float)
            success_grid = np.full_like(alpha_grid, np.nan, dtype=float)
            
            point_dict = {}
            for point in all_points:
                key = (point['alpha'], point['beta'])
                point_dict[key] = point
            
            for i in range(alpha_grid.shape[0]):
                for j in range(alpha_grid.shape[1]):
                    key = (alpha_grid[i, j], beta_grid[i, j])
                    if key in point_dict:
                        iterations_grid[i, j] = point_dict[key]['iterations']
                        success_grid[i, j] = 1 if point_dict[key]['success'] else 0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # График 1: Распределение точек по параметрам
    ax = axes[0]
    # ax.set_xlim(-0.15,0.1)
    scatter1 = ax.scatter(alphas, betas, c=iterations, cmap='viridis', s=50, alpha=0.7)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_title('Число итераций метода Ньютона')
    plt.colorbar(scatter1, ax=ax, label='Итерации')
    
    # График 2: Успешность сходимости
    ax = axes[1]
    # ax.set_xlim(-0.15,0.1)
    # Разделяем успешные и неуспешные точки
    success_mask = success_flags == 1
    if np.any(success_mask):
        ax.scatter(alphas[success_mask], betas[success_mask], 
                  c='green', s=50, alpha=0.7, label='Успех')
    if np.any(~success_mask):
        ax.scatter(alphas[~success_mask], betas[~success_mask], 
                  c='red', s=50, alpha=0.7, label='Неудача')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_title('Успешность сходимости')
    ax.legend()
    
    # График 3: 3D график итераций
    if 'iterations_grid' in locals() and not np.all(np.isnan(iterations_grid)):
        ax = axes[2]
        # ax.set_xlim(-0.15,0.1)
        contour = ax.contourf(alpha_grid, beta_grid, iterations_grid, 
                             levels=20, cmap='viridis')
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_title('Число итераций (интерполяция)')
        plt.colorbar(contour, ax=ax, label='Итерации')
    else:
        # Альтернативный график: гистограмма итераций
        ax = axes[2]
        ax.hist(iterations, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Число итераций')
        ax.set_ylabel('Частота')
        ax.set_title('Распределение числа итераций')
    
    plt.tight_layout()
    plt.show()
    
    # Выводим статистику
    print("\nСтатистика:")
    print(f"Всего точек: {len(all_points)}")
    print(f"Успешных: {np.sum(success_flags)}")
    print(f"Неудачных: {len(all_points) - np.sum(success_flags)}")
    if len(iterations) > 0:
        print(f"Среднее число итераций: {np.mean(iterations):.2f}")
        print(f"Максимальное число итераций: {np.max(iterations)}")


def main():
    # Размерность
    dim = 3

    # Начальные параметры
    alpha0 = -0.0250
    beta0 = 0.775
    
    x = (0.7 + beta0-1)/2 + np.sqrt(((0.7+beta0-1)**2)/4 + alpha0)
    x_start = np.array([x, x, x])  # Неподвижная точка

    # Диапазоны параметров
    alpha_min, alpha_max = -0.15, 0.1
    beta_min, beta_max = 0.7, 1
    
    # Шаги
    alpha_step = 0.001
    beta_step = 0.001
    
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
        for item in results['all_points']:
            alpha = item['alpha']
            beta = item['beta']
            coords = item['x']
           
            line_values = [alpha, beta] + coords.tolist()
            line = ", ".join(map(str, line_values))

            f.write(line + "\\n")

    # Визуализируем результаты
    visualize_results(results)


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from typing import Callable, Tuple, List

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class OneDimensionalOptimization:
    """一维优化方法集合类"""
    
    def __init__(self, tolerance=1e-6, max_iterations=1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.iteration_history = []
    
    def bisection_method(self, f: Callable, a: float, b: float) -> Tuple[float, int]:
        """
        二分法 - 适用于单峰函数
        
        参数:
        f: 目标函数
        a, b: 搜索区间
        
        返回:
        (最优点, 迭代次数)
        """
        self.iteration_history = []
        iteration = 0
        
        while (b - a) > self.tolerance and iteration < self.max_iterations:
            # 计算三分点
            x1 = a + (b - a) / 3
            x2 = a + 2 * (b - a) / 3
            
            self.iteration_history.append((a, b, (a + b) / 2))
            
            if f(x1) > f(x2):
                a = x1
            else:
                b = x2
            
            iteration += 1
        
        return (a + b) / 2, iteration
    
    def grid_search(self, f: Callable, a: float, b: float, n_points: int = 100) -> Tuple[float, int]:
        """
        格点法 - 在区间内均匀采样
        
        参数:
        f: 目标函数
        a, b: 搜索区间
        n_points: 格点数量
        
        返回:
        (最优点, 迭代次数)
        """
        self.iteration_history = []
        
        # 生成格点
        x_points = np.linspace(a, b, n_points)
        f_values = [f(x) for x in x_points]
        
        # 找到最小值点
        min_index = np.argmin(f_values)
        optimal_x = x_points[min_index]
        
        self.iteration_history = [(a, b, optimal_x)]
        
        return optimal_x, 1
    
    def fibonacci_method(self, f: Callable, a: float, b: float, n: int = 20) -> Tuple[float, int]:
        """
        斐波那契法
        
        参数:
        f: 目标函数
        a, b: 搜索区间
        n: 斐波那契数列项数
        
        返回:
        (最优点, 迭代次数)
        """
        self.iteration_history = []
        
        # 生成斐波那契数列
        fib = [1, 1]
        for i in range(2, n + 2):
            fib.append(fib[i-1] + fib[i-2])
        
        # 初始化
        L = b - a
        k = 0
        
        while k < n:
            # 计算内分点
            x1 = a + (fib[n-k-1] / fib[n-k+1]) * L
            x2 = a + (fib[n-k] / fib[n-k+1]) * L
            
            self.iteration_history.append((a, b, (x1 + x2) / 2))
            
            if f(x1) > f(x2):
                a = x1
            else:
                b = x2
            
            L = b - a
            k += 1
        
        return (a + b) / 2, k
    
    def golden_section_method(self, f: Callable, a: float, b: float) -> Tuple[float, int]:
        """
        黄金分割法
        
        参数:
        f: 目标函数
        a, b: 搜索区间
        
        返回:
        (最优点, 迭代次数)
        """
        self.iteration_history = []
        phi = (1 + np.sqrt(5)) / 2  # 黄金比例
        resphi = 2 - phi  # 1/phi
        
        # 计算初始内分点
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        f1, f2 = f(x1), f(x2)
        
        iteration = 0
        
        while abs(b - a) > self.tolerance and iteration < self.max_iterations:
            self.iteration_history.append((a, b, (a + b) / 2))
            
            if f1 > f2:
                a, x1, f1 = x1, x2, f2
                x2 = b - resphi * (b - a)
                f2 = f(x2)
            else:
                b, x2, f2 = x2, x1, f1
                x1 = a + resphi * (b - a)
                f1 = f(x1)
            
            iteration += 1
        
        return (a + b) / 2, iteration
    
    def quadratic_interpolation(self, f: Callable, x0: float, x1: float, x2: float) -> Tuple[float, int]:
        """
        二次插值法
        
        参数:
        f: 目标函数
        x0, x1, x2: 三个初始点
        
        返回:
        (最优点, 迭代次数)
        """
        self.iteration_history = []
        iteration = 0
        
        # 确保x0 < x1 < x2
        points = sorted([x0, x1, x2])
        x0, x1, x2 = points[0], points[1], points[2]
        
        while iteration < self.max_iterations:
            # 计算函数值
            f0, f1, f2 = f(x0), f(x1), f(x2)
            
            self.iteration_history.append((x0, x2, x1))
            
            # 二次插值计算新点
            numerator = (x1 - x0)**2 * (f1 - f2) - (x1 - x2)**2 * (f1 - f0)
            denominator = 2 * ((x1 - x0) * (f1 - f2) - (x1 - x2) * (f1 - f0))
            
            if abs(denominator) < 1e-12:
                break
            
            x_new = x1 - numerator / denominator
            f_new = f(x_new)
            
            # 检查收敛
            if abs(x_new - x1) < self.tolerance:
                return x_new, iteration + 1
            
            # 更新点
            if x_new < x1:
                if f_new < f1:
                    x2, x1 = x1, x_new
                else:
                    x0 = x_new
            else:
                if f_new < f1:
                    x0, x1 = x1, x_new
                else:
                    x2 = x_new
            
            iteration += 1
        
        return x1, iteration
    
    def newton_method(self, f: Callable, x0: float, h: float = 1e-5) -> Tuple[float, int]:
        """
        牛顿法 - 使用数值导数
        
        参数:
        f: 目标函数
        x0: 初始点
        h: 计算导数的步长
        
        返回:
        (最优点, 迭代次数)
        """
        self.iteration_history = []
        x = x0
        iteration = 0
        
        while iteration < self.max_iterations:
            # 计算一阶和二阶导数
            f_prime = derivative(f, x, dx=h)
            f_double_prime = derivative(lambda t: derivative(f, t, dx=h), x, dx=h)
            
            self.iteration_history.append((x - 1, x + 1, x))
            
            # 检查收敛
            if abs(f_prime) < self.tolerance:
                break
            
            if abs(f_double_prime) < 1e-12:
                print("警告：二阶导数接近零，牛顿法可能不收敛")
                break
            
            # 牛顿迭代
            x_new = x - f_prime / f_double_prime
            
            if abs(x_new - x) < self.tolerance:
                return x_new, iteration + 1
            
            x = x_new
            iteration += 1
        
        return x, iteration

def test_optimization_methods():
    """测试所有优化方法"""
    
    # 定义测试函数
    def f1(x):
        """二次函数 f(x) = (x-2)^2 + 1，最小值点在x=2"""
        return (x - 2)**2 + 1
    
    def f2(x):
        """四次函数 f(x) = x^4 - 4x^3 + 6x^2 - 4x + 1，最小值点在x=1"""
        return x**4 - 4*x**3 + 6*x**2 - 4*x + 1
    
    def f3(x):
        """组合函数 f(x) = sin(x) + x^2/4"""
        return np.sin(x) + x**2 / 4
    
    # 初始化优化器
    optimizer = OneDimensionalOptimization(tolerance=1e-6)
    
    # 测试函数列表
    test_functions = [
        (f1, "f(x) = (x-2)² + 1", -1, 5, 2),
        (f2, "f(x) = x⁴ - 4x³ + 6x² - 4x + 1", -1, 3, 1),
        (f3, "f(x) = sin(x) + x²/4", -3, 3, 0)
    ]
    
    methods = [
        ("二分法", optimizer.bisection_method),
        ("格点法", optimizer.grid_search),
        ("斐波那契法", optimizer.fibonacci_method),
        ("黄金分割法", optimizer.golden_section_method),
        ("二次插值法", lambda f, a, b: optimizer.quadratic_interpolation(f, a, (a+b)/2, b)),
        ("牛顿法", lambda f, a, b: optimizer.newton_method(f, (a+b)/2))
    ]
    
    print("=" * 80)
    print("一维优化方法比较测试")
    print("=" * 80)
    
    for i, (func, func_name, a, b, true_min) in enumerate(test_functions):
        print(f"\n测试函数 {i+1}: {func_name}")
        print(f"搜索区间: [{a}, {b}]")
        print(f"理论最优点: {true_min}")
        print("-" * 60)
        
        results = []
        
        for method_name, method in methods:
            try:
                if method_name in ["二次插值法", "牛顿法"]:
                    optimal_x, iterations = method(func, a, b)
                else:
                    optimal_x, iterations = method(func, a, b)
                
                optimal_f = func(optimal_x)
                error = abs(optimal_x - true_min)
                
                results.append({
                    'method': method_name,
                    'x': optimal_x,
                    'f': optimal_f,
                    'iterations': iterations,
                    'error': error
                })
                
                print(f"{method_name:10s}: x* = {optimal_x:8.6f}, f(x*) = {optimal_f:8.6f}, "
                      f"误差 = {error:8.6f}, 迭代次数 = {iterations:3d}")
                      
            except Exception as e:
                print(f"{method_name:10s}: 执行失败 - {str(e)}")
        
        # 绘制结果对比图
        plt.figure(figsize=(12, 8))
        
        # 绘制函数曲线
        x_plot = np.linspace(a, b, 1000)
        y_plot = [func(x) for x in x_plot]
        plt.plot(x_plot, y_plot, 'b-', linewidth=2, label=func_name)
        
        # 绘制各方法找到的最优点
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        for j, result in enumerate(results):
            plt.plot(result['x'], result['f'], 'o', color=colors[j % len(colors)], 
                    markersize=8, label=f"{result['method']}: ({result['x']:.4f}, {result['f']:.4f})")
        
        # 标记真实最优点
        plt.plot(true_min, func(true_min), 's', color='black', markersize=10, 
                label=f'真实最优点: ({true_min}, {func(true_min):.4f})')
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'测试函数 {i+1}: {func_name} - 优化结果比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def simple_example():
    """简单使用示例"""
    print("=" * 50)
    print("一维优化方法简单示例")
    print("=" * 50)
    
    # 定义一个简单的二次函数 f(x) = (x-3)^2 + 2
    def target_function(x):
        return (x - 3)**2 + 2
    
    print("目标函数: f(x) = (x-3)² + 2")
    print("理论最优点: x* = 3, f(x*) = 2")
    print("搜索区间: [-2, 8]")
    print()
    
    # 创建优化器
    optimizer = OneDimensionalOptimization(tolerance=1e-6)
    
    # 测试区间
    a, b = -2, 8
    
    # 1. 二分法
    x_opt, iter_count = optimizer.bisection_method(target_function, a, b)
    print(f"1. 二分法: x* = {x_opt:.6f}, f(x*) = {target_function(x_opt):.6f}, 迭代次数 = {iter_count}")
    
    # 2. 格点法
    x_opt, iter_count = optimizer.grid_search(target_function, a, b, n_points=50)
    print(f"2. 格点法: x* = {x_opt:.6f}, f(x*) = {target_function(x_opt):.6f}, 格点数 = 50")
    
    # 3. 斐波那契法
    x_opt, iter_count = optimizer.fibonacci_method(target_function, a, b, n=15)
    print(f"3. 斐波那契法: x* = {x_opt:.6f}, f(x*) = {target_function(x_opt):.6f}, 迭代次数 = {iter_count}")
    
    # 4. 黄金分割法
    x_opt, iter_count = optimizer.golden_section_method(target_function, a, b)
    print(f"4. 黄金分割法: x* = {x_opt:.6f}, f(x*) = {target_function(x_opt):.6f}, 迭代次数 = {iter_count}")
    
    # 5. 二次插值法
    x_opt, iter_count = optimizer.quadratic_interpolation(target_function, a, (a+b)/2, b)
    print(f"5. 二次插值法: x* = {x_opt:.6f}, f(x*) = {target_function(x_opt):.6f}, 迭代次数 = {iter_count}")
    
    # 6. 牛顿法
    x_opt, iter_count = optimizer.newton_method(target_function, (a+b)/2)
    print(f"6. 牛顿法: x* = {x_opt:.6f}, f(x*) = {target_function(x_opt):.6f}, 迭代次数 = {iter_count}")

if __name__ == "__main__":
    # 运行简单示例
    simple_example()
    
    print("\n" + "="*50)
    print("是否运行完整测试？(y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        test_optimization_methods()

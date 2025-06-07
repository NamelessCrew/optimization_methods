import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import line_search
from typing import Callable, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

class TwoDimensionalOptimization:
    """二维优化方法集合类"""
    
    def __init__(self, tolerance=1e-6, max_iterations=1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.path_history = []
    
    def numerical_gradient(self, f: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """计算数值梯度"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        return grad
    
    def numerical_hessian(self, f: Callable, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """计算数值海塞矩阵"""
        n = len(x)
        hessian = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += h
                x_pp[j] += h
                x_pm[i] += h
                x_pm[j] -= h
                x_mp[i] -= h
                x_mp[j] += h
                x_mm[i] -= h
                x_mm[j] -= h
                
                hessian[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h)
        
        return hessian
    
    def armijo_line_search(self, f: Callable, x: np.ndarray, direction: np.ndarray, 
                          alpha_init: float = 1.0, c1: float = 1e-4, rho: float = 0.5) -> float:
        """Armijo线搜索"""
        alpha = alpha_init
        fx = f(x)
        grad_fx = self.numerical_gradient(f, x)
        slope = np.dot(grad_fx, direction)
        
        while f(x + alpha * direction) > fx + c1 * alpha * slope:
            alpha *= rho
            if alpha < 1e-10:
                break
        
        return alpha
    
    def steepest_descent(self, f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        """
        最速下降法
        
        参数:
        f: 目标函数
        x0: 初始点
        
        返回:
        (最优点, 迭代次数, 路径历史)
        """
        x = x0.copy()
        path = [x.copy()]
        
        for iteration in range(self.max_iterations):
            # 计算梯度
            grad = self.numerical_gradient(f, x)
            
            # 检查收敛条件
            if np.linalg.norm(grad) < self.tolerance:
                break
            
            # 搜索方向为负梯度方向
            direction = -grad
            
            # 线搜索确定步长
            alpha = self.armijo_line_search(f, x, direction)
            
            # 更新点
            x = x + alpha * direction
            path.append(x.copy())
        
        return x, iteration + 1, path
    
    def coordinate_descent(self, f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        """
        坐标轮换法
        
        参数:
        f: 目标函数
        x0: 初始点
        
        返回:
        (最优点, 迭代次数, 路径历史)
        """
        x = x0.copy()
        path = [x.copy()]
        n = len(x)
        
        for iteration in range(self.max_iterations):
            x_old = x.copy()
            
            # 沿每个坐标轴方向进行一维搜索
            for i in range(n):
                # 坐标方向
                direction = np.zeros(n)
                direction[i] = 1.0
                
                # 计算梯度在该方向的分量
                grad = self.numerical_gradient(f, x)
                if abs(grad[i]) > self.tolerance:
                    direction[i] = -np.sign(grad[i])
                    
                    # 线搜索
                    alpha = self.armijo_line_search(f, x, direction)
                    x = x + alpha * direction
                    path.append(x.copy())
            
            # 检查收敛
            if np.linalg.norm(x - x_old) < self.tolerance:
                break
        
        return x, iteration + 1, path
    
    def conjugate_direction(self, f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        """
        共轭方向法
        
        参数:
        f: 目标函数
        x0: 初始点
        
        返回:
        (最优点, 迭代次数, 路径历史)
        """
        x = x0.copy()
        path = [x.copy()]
        n = len(x)
        
        # 初始化方向集合（使用单位向量）
        directions = np.eye(n)
        
        for iteration in range(self.max_iterations):
            x_start = x.copy()
            
            # 沿每个共轭方向进行搜索
            for i in range(n):
                grad = self.numerical_gradient(f, x)
                if np.linalg.norm(grad) < self.tolerance:
                    break
                
                direction = directions[i]
                alpha = self.armijo_line_search(f, x, direction)
                x = x + alpha * direction
                path.append(x.copy())
            
            # 更新方向集合（简化版本）
            if iteration > 0:
                # 使用Gram-Schmidt正交化
                new_direction = x - x_start
                if np.linalg.norm(new_direction) > 1e-10:
                    new_direction = new_direction / np.linalg.norm(new_direction)
                    directions = np.roll(directions, -1, axis=0)
                    directions[-1] = new_direction
            
            # 检查收敛
            if np.linalg.norm(x - x_start) < self.tolerance:
                break
        
        return x, iteration + 1, path
    
    def conjugate_gradient(self, f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        """
        共轭梯度法
        
        参数:
        f: 目标函数
        x0: 初始点
        
        返回:
        (最优点, 迭代次数, 路径历史)
        """
        x = x0.copy()
        path = [x.copy()]
        
        # 初始梯度和搜索方向
        grad = self.numerical_gradient(f, x)
        direction = -grad
        
        for iteration in range(self.max_iterations):
            # 检查收敛条件
            if np.linalg.norm(grad) < self.tolerance:
                break
            
            # 线搜索确定步长
            alpha = self.armijo_line_search(f, x, direction)
            
            # 更新点
            x_new = x + alpha * direction
            path.append(x_new.copy())
            
            # 计算新梯度
            grad_new = self.numerical_gradient(f, x_new)
            
            # 计算β (Polak-Ribiere公式)
            beta = max(0, np.dot(grad_new, grad_new - grad) / np.dot(grad, grad))
            
            # 更新搜索方向
            direction = -grad_new + beta * direction
            
            # 更新
            x = x_new
            grad = grad_new
        
        return x, iteration + 1, path
    
    def newton_method(self, f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        """
        牛顿法
        
        参数:
        f: 目标函数
        x0: 初始点
        
        返回:
        (最优点, 迭代次数, 路径历史)
        """
        x = x0.copy()
        path = [x.copy()]
        
        for iteration in range(self.max_iterations):
            # 计算梯度和海塞矩阵
            grad = self.numerical_gradient(f, x)
            hessian = self.numerical_hessian(f, x)
            
            # 检查收敛条件
            if np.linalg.norm(grad) < self.tolerance:
                break
            
            try:
                # 计算牛顿方向
                direction = -np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                # 如果海塞矩阵奇异，使用最速下降方向
                direction = -grad
            
            # 更新点
            x = x + direction
            path.append(x.copy())
        
        return x, iteration + 1, path
    
    def damped_newton_method(self, f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        """
        阻尼牛顿法
        
        参数:
        f: 目标函数
        x0: 初始点
        
        返回:
        (最优点, 迭代次数, 路径历史)
        """
        x = x0.copy()
        path = [x.copy()]
        
        for iteration in range(self.max_iterations):
            # 计算梯度和海塞矩阵
            grad = self.numerical_gradient(f, x)
            hessian = self.numerical_hessian(f, x)
            
            # 检查收敛条件
            if np.linalg.norm(grad) < self.tolerance:
                break
            
            try:
                # 计算牛顿方向
                direction = -np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                # 如果海塞矩阵奇异，使用最速下降方向
                direction = -grad
            
            # 线搜索确定步长（阻尼）
            alpha = self.armijo_line_search(f, x, direction)
            
            # 更新点
            x = x + alpha * direction
            path.append(x.copy())
        
        return x, iteration + 1, path
    
    def dfp_method(self, f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        """
        拟牛顿法 - DFP算法
        
        参数:
        f: 目标函数
        x0: 初始点
        
        返回:
        (最优点, 迭代次数, 路径历史)
        """
        x = x0.copy()
        path = [x.copy()]
        n = len(x)
        
        # 初始化逆海塞矩阵近似为单位矩阵
        H = np.eye(n)
        
        # 计算初始梯度
        grad = self.numerical_gradient(f, x)
        
        for iteration in range(self.max_iterations):
            # 检查收敛条件
            if np.linalg.norm(grad) < self.tolerance:
                break
            
            # 计算搜索方向
            direction = -np.dot(H, grad)
            
            # 线搜索确定步长
            alpha = self.armijo_line_search(f, x, direction)
            
            # 更新点
            x_new = x + alpha * direction
            path.append(x_new.copy())
            
            # 计算新梯度
            grad_new = self.numerical_gradient(f, x_new)
            
            # DFP更新公式
            s = x_new - x  # 位置差
            y = grad_new - grad  # 梯度差
            
            if np.dot(s, y) > 1e-10:  # 确保正定性
                # DFP公式更新逆海塞矩阵
                Hy = np.dot(H, y)
                H = H + np.outer(s, s) / np.dot(s, y) - np.outer(Hy, Hy) / np.dot(y, Hy)
            
            # 更新
            x = x_new
            grad = grad_new
        
        return x, iteration + 1, path

def test_optimization_methods():
    """测试所有二维优化方法"""
    
    # 定义测试函数
    def rosenbrock(x):
        """Rosenbrock函数: f(x,y) = 100(y-x²)² + (1-x)²，最小值点在(1,1)"""
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    def quadratic(x):
        """二次函数: f(x,y) = (x-1)² + (y-2)²，最小值点在(1,2)"""
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    def himmelblau(x):
        """Himmelblau函数: f(x,y) = (x²+y-11)² + (x+y²-7)²"""
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    
    # 测试函数列表
    test_functions = [
        (quadratic, "二次函数: f(x,y) = (x-1)² + (y-2)²", np.array([0.0, 0.0]), np.array([1.0, 2.0])),
        (rosenbrock, "Rosenbrock函数", np.array([-1.0, 1.0]), np.array([1.0, 1.0])),
        (himmelblau, "Himmelblau函数", np.array([0.0, 0.0]), np.array([3.0, 2.0]))
    ]
    
    # 初始化优化器
    optimizer = TwoDimensionalOptimization(tolerance=1e-6, max_iterations=100)
    
    # 优化方法列表
    methods = [
        ("最速下降法", optimizer.steepest_descent),
        ("坐标轮换法", optimizer.coordinate_descent),
        ("共轭方向法", optimizer.conjugate_direction),
        ("共轭梯度法", optimizer.conjugate_gradient),
        ("牛顿法", optimizer.newton_method),
        ("阻尼牛顿法", optimizer.damped_newton_method),
        ("拟牛顿法(DFP)", optimizer.dfp_method)
    ]
    
    print("=" * 80)
    print("二维优化方法比较测试")
    print("=" * 80)
    
    for func_idx, (func, func_name, x0, true_min) in enumerate(test_functions):
        print(f"\n测试函数 {func_idx + 1}: {func_name}")
        print(f"初始点: ({x0[0]:.2f}, {x0[1]:.2f})")
        print(f"理论最优点: ({true_min[0]:.2f}, {true_min[1]:.2f})")
        print("-" * 60)
        
        # 创建等高线图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # 准备等高线数据
        x_range = np.linspace(x0[0] - 3, x0[0] + 5, 100)
        y_range = np.linspace(x0[1] - 3, x0[1] + 5, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
        
        results = []
        
        for method_idx, (method_name, method) in enumerate(methods):
            try:
                optimal_x, iterations, path = method(func, x0)
                optimal_f = func(optimal_x)
                error = np.linalg.norm(optimal_x - true_min)
                
                results.append({
                    'method': method_name,
                    'x': optimal_x,
                    'f': optimal_f,
                    'iterations': iterations,
                    'error': error,
                    'path': path
                })
                
                print(f"{method_name:15s}: x* = ({optimal_x[0]:7.4f}, {optimal_x[1]:7.4f}), "
                      f"f(x*) = {optimal_f:8.4f}, 误差 = {error:7.4f}, 迭代次数 = {iterations:3d}")
                
                # 绘制优化路径
                ax = axes[method_idx]
                contour = ax.contour(X, Y, Z, levels=20, alpha=0.6)
                ax.clabel(contour, inline=True, fontsize=8)
                
                # 绘制优化路径
                path_array = np.array(path)
                ax.plot(path_array[:, 0], path_array[:, 1], 'ro-', markersize=4, linewidth=1.5)
                ax.plot(x0[0], x0[1], 'go', markersize=8, label='起始点')
                ax.plot(optimal_x[0], optimal_x[1], 'r*', markersize=12, label='最优点')
                ax.plot(true_min[0], true_min[1], 'bs', markersize=8, label='理论最优')
                
                ax.set_title(f'{method_name}\n迭代次数: {iterations}, 误差: {error:.4f}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"{method_name:15s}: 执行失败 - {str(e)}")
                
                # 空白图
                ax = axes[method_idx]
                ax.text(0.5, 0.5, f'{method_name}\n执行失败', 
                        transform=ax.transAxes, ha='center', va='center')
                ax.set_title(method_name)
        
        # 删除多余的子图
        for i in range(len(methods), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.suptitle(f'测试函数 {func_idx + 1}: {func_name}', fontsize=16, y=0.98)
        plt.show()

def simple_example():
    """简单使用示例"""
    print("=" * 50)
    print("二维优化方法简单示例")
    print("=" * 50)
    
    # 定义一个简单的二次函数
    def target_function(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2
    
    print("目标函数: f(x,y) = (x-2)² + (y-1)²")
    print("理论最优点: (2, 1)")
    print("初始点: (0, 0)")
    print()
    
    # 创建优化器
    optimizer = TwoDimensionalOptimization(tolerance=1e-6, max_iterations=50)
    
    # 初始点
    x0 = np.array([0.0, 0.0])
    
    # 测试所有方法
    methods = [
        ("最速下降法", optimizer.steepest_descent),
        ("坐标轮换法", optimizer.coordinate_descent),
        ("共轭方向法", optimizer.conjugate_direction),
        ("共轭梯度法", optimizer.conjugate_gradient),
        ("牛顿法", optimizer.newton_method),
        ("阻尼牛顿法", optimizer.damped_newton_method),
        ("拟牛顿法(DFP)", optimizer.dfp_method)
    ]
    
    for method_name, method in methods:
        try:
            x_opt, iterations, path = method(target_function, x0)
            f_opt = target_function(x_opt)
            print(f"{method_name:15s}: x* = ({x_opt[0]:.6f}, {x_opt[1]:.6f}), "
                  f"f(x*) = {f_opt:.6f}, 迭代次数 = {iterations}")
        except Exception as e:
            print(f"{method_name:15s}: 执行失败 - {str(e)}")

if __name__ == "__main__":
    # 运行简单示例
    simple_example()
    
    print("\n" + "="*50)
    print("是否运行完整测试和可视化？(y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        test_optimization_methods()

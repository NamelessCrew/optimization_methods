import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
from typing import Callable, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ConstrainedOptimization:
    """带约束的优化方法集合类"""
    
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
    
    def is_feasible(self, x: np.ndarray, eq_constraints: List[Callable] = None, 
                   ineq_constraints: List[Callable] = None) -> bool:
        """检查点是否满足约束条件"""
        if eq_constraints:
            for constraint in eq_constraints:
                if abs(constraint(x)) > self.tolerance:
                    return False
        
        if ineq_constraints:
            for constraint in ineq_constraints:
                if constraint(x) < -self.tolerance:
                    return False
        
        return True
    
    def project_to_feasible(self, x: np.ndarray, eq_constraints: List[Callable] = None, 
                           ineq_constraints: List[Callable] = None) -> np.ndarray:
        """将点投影到可行域"""
        if not eq_constraints and not ineq_constraints:
            return x
        
        def objective(x_proj):
            return np.sum((x_proj - x)**2)
        
        def eq_constraints_func(x_proj):
            if not eq_constraints:
                return []
            return [constraint(x_proj) for constraint in eq_constraints]
        
        def ineq_constraints_func(x_proj):
            if not ineq_constraints:
                return []
            return [constraint(x_proj) for constraint in ineq_constraints]
        
        try:
            from scipy.optimize import minimize
            
            constraints = []
            if eq_constraints:
                constraints.append({'type': 'eq', 'fun': lambda x_proj: eq_constraints_func(x_proj)})
            if ineq_constraints:
                constraints.append({'type': 'ineq', 'fun': lambda x_proj: ineq_constraints_func(x_proj)})
            
            result = minimize(objective, x, constraints=constraints, method='SLSQP')
            if result.success:
                return result.x
            else:
                return x
        except:
            return x
    
    def constrained_coordinate_descent(self, f: Callable, x0: np.ndarray, 
                                     eq_constraints: List[Callable] = None,
                                     ineq_constraints: List[Callable] = None) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        """
        约束坐标轮换法
        
        参数:
        f: 目标函数
        x0: 初始点
        eq_constraints: 等式约束列表 [g1(x), g2(x), ...] = 0
        ineq_constraints: 不等式约束列表 [h1(x), h2(x), ...] >= 0
        
        返回:
        (最优点, 迭代次数, 路径历史)
        """
        x = x0.copy()
        path = [x.copy()]
        n = len(x)
        
        # 确保初始点可行
        x = self.project_to_feasible(x, eq_constraints, ineq_constraints)
        path.append(x.copy())
        
        for iteration in range(self.max_iterations):
            x_old = x.copy()
            
            # 沿每个坐标轴方向进行一维搜索
            for i in range(n):
                # 定义沿坐标轴i的一维函数
                def line_function(alpha):
                    x_temp = x.copy()
                    x_temp[i] += alpha
                    
                    # 检查约束
                    if not self.is_feasible(x_temp, eq_constraints, ineq_constraints):
                        return float('inf')
                    
                    return f(x_temp)
                
                # 在满足约束的情况下，寻找最优步长
                best_alpha = 0
                best_value = f(x)
                
                # 搜索正负方向
                step_sizes = [0.1, 0.01, 0.001]
                for step in step_sizes:
                    for direction in [1, -1]:
                        for k in range(1, 21):  # 最多尝试20步
                            alpha = direction * k * step
                            x_temp = x.copy()
                            x_temp[i] += alpha
                            
                            if self.is_feasible(x_temp, eq_constraints, ineq_constraints):
                                value = f(x_temp)
                                if value < best_value:
                                    best_value = value
                                    best_alpha = alpha
                            else:
                                break  # 超出可行域，停止在这个方向搜索
                
                # 更新坐标
                if best_alpha != 0:
                    x[i] += best_alpha
                    path.append(x.copy())
            
            # 检查收敛
            if np.linalg.norm(x - x_old) < self.tolerance:
                break
        
        return x, iteration + 1, path
    
    def lagrange_method(self, f: Callable, x0: np.ndarray, 
                       eq_constraints: List[Callable] = None,
                       lambda0: np.ndarray = None) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        """
        拉格朗日法（仅处理等式约束）
        
        参数:
        f: 目标函数
        x0: 初始点
        eq_constraints: 等式约束列表
        lambda0: 拉格朗日乘数初始值
        
        返回:
        (最优点, 迭代次数, 路径历史)
        """
        if not eq_constraints:
            # 无约束优化，退化为梯度下降
            from two_dim import TwoDimensionalOptimization
            optimizer = TwoDimensionalOptimization(self.tolerance, self.max_iterations)
            return optimizer.steepest_descent(f, x0)
        
        x = x0.copy()
        path = [x.copy()]
        
        # 初始化拉格朗日乘数
        if lambda0 is None:
            lambda_multipliers = np.zeros(len(eq_constraints))
        else:
            lambda_multipliers = lambda0.copy()
        
        for iteration in range(self.max_iterations):
            # 构造拉格朗日函数的梯度
            def lagrangian_gradient(vars):
                x_vars = vars[:len(x)]
                lambda_vars = vars[len(x):]
                
                # 目标函数梯度
                grad_f = self.numerical_gradient(f, x_vars)
                
                # 约束函数梯度
                for i, constraint in enumerate(eq_constraints):
                    grad_g = self.numerical_gradient(constraint, x_vars)
                    grad_f += lambda_vars[i] * grad_g
                
                # 约束值（KKT条件）
                constraint_values = np.array([constraint(x_vars) for constraint in eq_constraints])
                
                return np.concatenate([grad_f, constraint_values])
            
            # 求解KKT条件
            variables = np.concatenate([x, lambda_multipliers])
            
            try:
                # 使用牛顿法求解非线性方程组
                from scipy.optimize import fsolve
                solution = fsolve(lagrangian_gradient, variables)
                
                x_new = solution[:len(x)]
                lambda_multipliers = solution[len(x):]
                
                # 检查收敛
                if np.linalg.norm(x_new - x) < self.tolerance:
                    x = x_new
                    path.append(x.copy())
                    break
                
                x = x_new
                path.append(x.copy())
                
            except:
                # 如果求解失败，使用梯度下降
                grad = self.numerical_gradient(f, x)
                
                # 添加约束项的梯度
                for i, constraint in enumerate(eq_constraints):
                    grad_g = self.numerical_gradient(constraint, x)
                    grad += lambda_multipliers[i] * grad_g
                
                # 简单的步长选择
                alpha = 0.01
                x = x - alpha * grad
                path.append(x.copy())
                
                # 更新拉格朗日乘数
                for i, constraint in enumerate(eq_constraints):
                    lambda_multipliers[i] += 0.1 * constraint(x)
        
        return x, iteration + 1, path
    
    def penalty_method(self, f: Callable, x0: np.ndarray,
                      eq_constraints: List[Callable] = None,
                      ineq_constraints: List[Callable] = None,
                      penalty_param: float = 1.0,
                      penalty_increase: float = 10.0) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        """
        惩罚函数法
        
        参数:
        f: 目标函数
        x0: 初始点
        eq_constraints: 等式约束列表
        ineq_constraints: 不等式约束列表
        penalty_param: 初始惩罚参数
        penalty_increase: 惩罚参数增长因子
        
        返回:
        (最优点, 迭代次数, 路径历史)
        """
        x = x0.copy()
        path = [x.copy()]
        rho = penalty_param  # 惩罚参数
        
        # 导入无约束优化器
        from two_dim import TwoDimensionalOptimization
        optimizer = TwoDimensionalOptimization(self.tolerance, 50)  # 每次内层迭代较少
        
        for outer_iteration in range(20):  # 外层迭代
            # 构造惩罚函数
            def penalty_function(x_var):
                objective_value = f(x_var)
                penalty_value = 0
                
                # 等式约束惩罚项
                if eq_constraints:
                    for constraint in eq_constraints:
                        penalty_value += rho * constraint(x_var)**2
                
                # 不等式约束惩罚项
                if ineq_constraints:
                    for constraint in ineq_constraints:
                        violation = max(0, -constraint(x_var))  # 只惩罚违反约束的部分
                        penalty_value += rho * violation**2
                
                return objective_value + penalty_value
            
            # 无约束优化惩罚函数
            try:
                x_new, inner_iterations, inner_path = optimizer.steepest_descent(penalty_function, x)
                
                # 将内层路径添加到总路径
                path.extend(inner_path[1:])  # 跳过第一个点，避免重复
                
                # 检查收敛
                if np.linalg.norm(x_new - x) < self.tolerance:
                    x = x_new
                    break
                
                # 检查约束满足情况
                constraint_violation = 0
                if eq_constraints:
                    for constraint in eq_constraints:
                        constraint_violation += abs(constraint(x_new))
                
                if ineq_constraints:
                    for constraint in ineq_constraints:
                        constraint_violation += max(0, -constraint(x_new))
                
                # 如果约束满足得足够好，可以停止
                if constraint_violation < self.tolerance:
                    x = x_new
                    break
                
                x = x_new
                rho *= penalty_increase  # 增加惩罚参数
                
            except Exception as e:
                print(f"惩罚函数法内层优化失败: {e}")
                break
        
        return x, outer_iteration + 1, path

def define_test_problems():
    """定义测试问题"""
    
    problems = {}
    
    # 问题1：简单二次规划
    def problem1():
        def objective(x):
            return (x[0] - 2)**2 + (x[1] - 1)**2
        
        def eq_constraint1(x):
            return x[0] + x[1] - 3  # x + y = 3
        
        def ineq_constraint1(x):
            return x[0]  # x >= 0
        
        def ineq_constraint2(x):
            return x[1]  # y >= 0
        
        return {
            'objective': objective,
            'eq_constraints': [eq_constraint1],
            'ineq_constraints': [ineq_constraint1, ineq_constraint2],
            'x0': np.array([1.0, 1.0]),
            'optimal': np.array([1.5, 1.5]),
            'name': '二次规划: min (x-2)²+(y-1)², s.t. x+y=3, x≥0, y≥0'
        }
    
    # 问题2：圆形约束
    def problem2():
        def objective(x):
            return x[0]**2 + x[1]**2
        
        def ineq_constraint1(x):
            return 4 - (x[0] - 1)**2 - (x[1] - 1)**2  # (x-1)² + (y-1)² <= 4
        
        return {
            'objective': objective,
            'eq_constraints': None,
            'ineq_constraints': [ineq_constraint1],
            'x0': np.array([0.0, 0.0]),
            'optimal': np.array([0.0, 0.0]),
            'name': '圆形约束: min x²+y², s.t. (x-1)²+(y-1)²≤4'
        }
    
    # 问题3：等式和不等式约束组合
    def problem3():
        def objective(x):
            return (x[0] - 3)**2 + (x[1] - 2)**2
        
        def eq_constraint1(x):
            return x[0]**2 + x[1]**2 - 5  # x² + y² = 5
        
        def ineq_constraint1(x):
            return x[0] + x[1] - 1  # x + y >= 1
        
        return {
            'objective': objective,
            'eq_constraints': [eq_constraint1],
            'ineq_constraints': [ineq_constraint1],
            'x0': np.array([1.0, 2.0]),
            'optimal': np.array([1.0, 2.0]),  # 近似最优解
            'name': '组合约束: min (x-3)²+(y-2)², s.t. x²+y²=5, x+y≥1'
        }
    
    problems['problem1'] = problem1()
    problems['problem2'] = problem2()
    problems['problem3'] = problem3()
    
    return problems

def test_constrained_optimization():
    """测试所有约束优化方法"""
    
    print("=" * 80)
    print("带约束优化方法比较测试")
    print("=" * 80)
    
    # 获取测试问题
    problems = define_test_problems()
    
    # 创建优化器
    optimizer = ConstrainedOptimization(tolerance=1e-6, max_iterations=100)
    
    for problem_name, problem in problems.items():
        print(f"\n{problem['name']}")
        print("-" * 60)
        
        objective = problem['objective']
        eq_constraints = problem['eq_constraints']
        ineq_constraints = problem['ineq_constraints']
        x0 = problem['x0']
        
        print(f"初始点: ({x0[0]:.2f}, {x0[1]:.2f})")
        print(f"初始函数值: {objective(x0):.6f}")
        
        # 检查初始点是否可行
        is_feasible_initial = optimizer.is_feasible(x0, eq_constraints, ineq_constraints)
        print(f"初始点可行性: {'是' if is_feasible_initial else '否'}")
        print()
        
        results = []
        
        # 方法1：约束坐标轮换法
        try:
            print("1. 约束坐标轮换法:")
            x_opt, iterations, path = optimizer.constrained_coordinate_descent(
                objective, x0, eq_constraints, ineq_constraints)
            
            final_value = objective(x_opt)
            is_feasible_final = optimizer.is_feasible(x_opt, eq_constraints, ineq_constraints)
            
            results.append({
                'method': '约束坐标轮换法',
                'x_opt': x_opt,
                'f_opt': final_value,
                'iterations': iterations,
                'feasible': is_feasible_final,
                'path': path
            })
            
            print(f"   结果: x* = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
            print(f"   函数值: f(x*) = {final_value:.6f}")
            print(f"   迭代次数: {iterations}")
            print(f"   最终可行性: {'是' if is_feasible_final else '否'}")
            
        except Exception as e:
            print(f"   执行失败: {str(e)}")
        
        # 方法2：拉格朗日法（仅用于有等式约束的问题）
        if eq_constraints:
            try:
                print("\n2. 拉格朗日法:")
                x_opt, iterations, path = optimizer.lagrange_method(
                    objective, x0, eq_constraints)
                
                final_value = objective(x_opt)
                is_feasible_final = optimizer.is_feasible(x_opt, eq_constraints, ineq_constraints)
                
                results.append({
                    'method': '拉格朗日法',
                    'x_opt': x_opt,
                    'f_opt': final_value,
                    'iterations': iterations,
                    'feasible': is_feasible_final,
                    'path': path
                })
                
                print(f"   结果: x* = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
                print(f"   函数值: f(x*) = {final_value:.6f}")
                print(f"   迭代次数: {iterations}")
                print(f"   最终可行性: {'是' if is_feasible_final else '否'}")
                
            except Exception as e:
                print(f"   执行失败: {str(e)}")
        else:
            print("\n2. 拉格朗日法: 跳过（无等式约束）")
        
        # 方法3：惩罚函数法
        try:
            print("\n3. 惩罚函数法:")
            x_opt, iterations, path = optimizer.penalty_method(
                objective, x0, eq_constraints, ineq_constraints)
            
            final_value = objective(x_opt)
            is_feasible_final = optimizer.is_feasible(x_opt, eq_constraints, ineq_constraints)
            
            results.append({
                'method': '惩罚函数法',
                'x_opt': x_opt,
                'f_opt': final_value,
                'iterations': iterations,
                'feasible': is_feasible_final,
                'path': path
            })
            
            print(f"   结果: x* = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
            print(f"   函数值: f(x*) = {final_value:.6f}")
            print(f"   外层迭代次数: {iterations}")
            print(f"   最终可行性: {'是' if is_feasible_final else '否'}")
            
        except Exception as e:
            print(f"   执行失败: {str(e)}")
        
        # 总结最佳结果
        if results:
            feasible_results = [r for r in results if r['feasible']]
            if feasible_results:
                best_result = min(feasible_results, key=lambda x: x['f_opt'])
                print(f"\n最佳可行解: {best_result['method']}")
                print(f"函数值: {best_result['f_opt']:.6f}")
                print(f"解: ({best_result['x_opt'][0]:.6f}, {best_result['x_opt'][1]:.6f})")
            else:
                print("\n警告: 所有方法都未找到可行解！")
        
        print("\n" + "="*60)

def visualize_constrained_optimization():
    """可视化约束优化问题"""
    
    print("约束优化可视化")
    print("="*50)
    
    # 选择一个问题进行可视化
    problems = define_test_problems()
    problem = problems['problem1']  # 选择第一个问题
    
    objective = problem['objective']
    eq_constraints = problem['eq_constraints']
    ineq_constraints = problem['ineq_constraints']
    x0 = problem['x0']
    
    # 创建优化器
    optimizer = ConstrainedOptimization(tolerance=1e-6, max_iterations=50)
    
    # 创建网格用于绘图
    x_range = np.linspace(-1, 4, 100)
    y_range = np.linspace(-1, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    # 计算目标函数值
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective(np.array([X[i, j], Y[i, j]]))
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = [
        ("约束坐标轮换法", lambda: optimizer.constrained_coordinate_descent(
            objective, x0, eq_constraints, ineq_constraints)),
        ("拉格朗日法", lambda: optimizer.lagrange_method(
            objective, x0, eq_constraints)),
        ("惩罚函数法", lambda: optimizer.penalty_method(
            objective, x0, eq_constraints, ineq_constraints))
    ]
    
    for idx, (method_name, method_func) in enumerate(methods):
        ax = axes[idx]
        
        # 绘制等高线
        contour = ax.contour(X, Y, Z, levels=20, alpha=0.6, colors='blue')
        ax.clabel(contour, inline=True, fontsize=8)
        
        # 绘制约束
        if eq_constraints:
            # 等式约束: x + y = 3
            x_eq = np.linspace(-1, 4, 100)
            y_eq = 3 - x_eq
            ax.plot(x_eq, y_eq, 'r-', linewidth=2, label='等式约束: x+y=3')
        
        if ineq_constraints:
            # 不等式约束: x >= 0, y >= 0
            ax.axhline(y=0, color='green', linestyle='--', linewidth=1, label='y≥0')
            ax.axvline(x=0, color='green', linestyle='--', linewidth=1, label='x≥0')
            
            # 填充可行域
            feasible_x = np.linspace(0, 4, 100)
            feasible_y = 3 - feasible_x
            feasible_y = np.maximum(feasible_y, 0)
            ax.fill_between(feasible_x, 0, feasible_y, alpha=0.2, color='green', label='可行域')
        
        try:
            # 运行优化方法
            x_opt, iterations, path = method_func()
            
            # 绘制优化路径
            if len(path) > 1:
                path_array = np.array(path)
                ax.plot(path_array[:, 0], path_array[:, 1], 'ko-', 
                       markersize=4, linewidth=2, alpha=0.8, label='优化路径')
            
            # 标记起始点和最优点
            ax.plot(x0[0], x0[1], 'go', markersize=10, label='起始点')
            ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15, label='最优点')
            
            is_feasible = optimizer.is_feasible(x_opt, eq_constraints, ineq_constraints)
            ax.set_title(f'{method_name}\n迭代次数: {iterations}, 可行: {"是" if is_feasible else "否"}')
            
            print(f"{method_name}: 找到解 ({x_opt[0]:.4f}, {x_opt[1]:.4f}), "
                  f"函数值 = {objective(x_opt):.6f}")
            
        except Exception as e:
            ax.text(0.5, 0.5, f'{method_name}\n执行失败', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            print(f"{method_name}: 执行失败 - {str(e)}")
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 4)
        ax.set_ylim(-0.5, 4)
    
    plt.tight_layout()
    plt.suptitle(f'约束优化问题可视化: {problem["name"]}', fontsize=14, y=1.02)
    plt.show()

def simple_example():
    """简单使用示例"""
    print("=" * 60)
    print("带约束优化方法简单示例")
    print("=" * 60)
    
    # 定义一个简单的约束优化问题
    def objective(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2
    
    def equality_constraint(x):
        return x[0] + x[1] - 3  # x + y = 3
    
    def inequality_constraint1(x):
        return x[0]  # x >= 0
    
    def inequality_constraint2(x):
        return x[1]  # y >= 0
    
    print("问题定义:")
    print("目标函数: min (x-2)² + (y-1)²")
    print("等式约束: x + y = 3")
    print("不等式约束: x >= 0, y >= 0")
    print("初始点: (1, 1)")
    print()
    
    # 创建优化器
    optimizer = ConstrainedOptimization(tolerance=1e-6, max_iterations=100)
    
    x0 = np.array([1.0, 1.0])
    eq_constraints = [equality_constraint]
    ineq_constraints = [inequality_constraint1, inequality_constraint2]
    
    print("方法比较:")
    print("-" * 40)
    
    # 1. 约束坐标轮换法
    try:
        x_opt, iterations, path = optimizer.constrained_coordinate_descent(
            objective, x0, eq_constraints, ineq_constraints)
        print(f"约束坐标轮换法: x*=({x_opt[0]:.4f}, {x_opt[1]:.4f}), "
              f"f(x*)={objective(x_opt):.6f}, 迭代={iterations}")
    except Exception as e:
        print(f"约束坐标轮换法: 失败 - {str(e)}")
    
    # 2. 拉格朗日法
    try:
        x_opt, iterations, path = optimizer.lagrange_method(
            objective, x0, eq_constraints)
        print(f"拉格朗日法      : x*=({x_opt[0]:.4f}, {x_opt[1]:.4f}), "
              f"f(x*)={objective(x_opt):.6f}, 迭代={iterations}")
    except Exception as e:
        print(f"拉格朗日法      : 失败 - {str(e)}")
    
    # 3. 惩罚函数法
    try:
        x_opt, iterations, path = optimizer.penalty_method(
            objective, x0, eq_constraints, ineq_constraints)
        print(f"惩罚函数法      : x*=({x_opt[0]:.4f}, {x_opt[1]:.4f}), "
              f"f(x*)={objective(x_opt):.6f}, 外层迭代={iterations}")
    except Exception as e:
        print(f"惩罚函数法      : 失败 - {str(e)}")

if __name__ == "__main__":
    # 运行简单示例
    simple_example()
    
    print("\n" + "="*60)
    print("是否运行完整测试？(y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        test_constrained_optimization()
    
    print("\n" + "="*60)
    print("是否运行可视化示例？(需要显示图形) (y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        visualize_constrained_optimization()
    
    print("\n程序结束！")

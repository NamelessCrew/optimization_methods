#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二维优化方法使用示例
演示如何使用七种不同的优化算法来优化二元函数
"""

import numpy as np
import matplotlib.pyplot as plt
from two_dim import TwoDimensionalOptimization

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def define_custom_function():
    """定义您自己的目标函数"""
    
    # 示例1：简单二次函数
    def quadratic_function(x):
        """f(x,y) = (x-a)² + (y-b)²"""
        a, b = 3, -1  # 可以修改这些参数
        return (x[0] - a)**2 + (x[1] - b)**2
    
    # 示例2：Rosenbrock函数（经典测试函数）
    def rosenbrock_function(x):
        """f(x,y) = 100(y-x²)² + (1-x)²"""
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    # 示例3：椭圆函数
    def elliptic_function(x):
        """f(x,y) = ax² + by²"""
        a, b = 2, 0.5  # 可以修改这些参数
        return a * x[0]**2 + b * x[1]**2
    
    # 示例4：带交叉项的二次函数
    def cross_term_function(x):
        """f(x,y) = x² + xy + y² - 4x - 3y + 5"""
        return x[0]**2 + x[0]*x[1] + x[1]**2 - 4*x[0] - 3*x[1] + 5
    
    return {
        '二次函数': (quadratic_function, np.array([3.0, -1.0])),
        'Rosenbrock函数': (rosenbrock_function, np.array([1.0, 1.0])),
        '椭圆函数': (elliptic_function, np.array([0.0, 0.0])),
        '带交叉项函数': (cross_term_function, np.array([1.0, 2.0]))
    }

def single_method_example():
    """单个方法使用示例"""
    print("=" * 60)
    print("单个优化方法使用示例")
    print("=" * 60)
    
    # 创建优化器
    optimizer = TwoDimensionalOptimization(tolerance=1e-8, max_iterations=100)
    
    # 定义目标函数
    def objective_function(x):
        return (x[0] - 2)**2 + (x[1] + 1)**2
    
    # 设置初始点
    x0 = np.array([0.0, 0.0])
    
    print(f"目标函数: f(x,y) = (x-2)² + (y+1)²")
    print(f"理论最优点: (2, -1)")
    print(f"初始点: ({x0[0]}, {x0[1]})")
    print()
    
    # 使用最速下降法
    print("1. 最速下降法:")
    x_opt, iterations, path = optimizer.steepest_descent(objective_function, x0)
    print(f"   结果: x* = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
    print(f"   函数值: f(x*) = {objective_function(x_opt):.8f}")
    print(f"   迭代次数: {iterations}")
    print(f"   路径长度: {len(path)}")
    
    # 使用牛顿法
    print("\n2. 牛顿法:")
    x_opt, iterations, path = optimizer.newton_method(objective_function, x0)
    print(f"   结果: x* = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
    print(f"   函数值: f(x*) = {objective_function(x_opt):.8f}")
    print(f"   迭代次数: {iterations}")
    print(f"   路径长度: {len(path)}")
    
    # 使用拟牛顿法(DFP)
    print("\n3. 拟牛顿法(DFP):")
    x_opt, iterations, path = optimizer.dfp_method(objective_function, x0)
    print(f"   结果: x* = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
    print(f"   函数值: f(x*) = {objective_function(x_opt):.8f}")
    print(f"   迭代次数: {iterations}")
    print(f"   路径长度: {len(path)}")

def compare_all_methods():
    """比较所有方法的性能"""
    print("\n" + "=" * 60)
    print("所有方法性能比较")
    print("=" * 60)
    
    # 创建优化器
    optimizer = TwoDimensionalOptimization(tolerance=1e-6, max_iterations=200)
    
    # 获取测试函数
    functions = define_custom_function()
    
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
    
    for func_name, (func, true_optimum) in functions.items():
        print(f"\n测试函数: {func_name}")
        print(f"理论最优点: ({true_optimum[0]:.2f}, {true_optimum[1]:.2f})")
        
        # 选择合适的初始点
        if func_name == 'Rosenbrock函数':
            x0 = np.array([-1.0, 1.0])
        else:
            x0 = np.array([0.0, 0.0])
        
        print(f"初始点: ({x0[0]:.2f}, {x0[1]:.2f})")
        print("-" * 50)
        
        results = []
        
        for method_name, method in methods:
            try:
                x_opt, iterations, path = method(func, x0)
                f_opt = func(x_opt)
                error = np.linalg.norm(x_opt - true_optimum)
                
                results.append({
                    'method': method_name,
                    'x_opt': x_opt,
                    'f_opt': f_opt,
                    'iterations': iterations,
                    'error': error,
                    'path_length': len(path)
                })
                
                print(f"{method_name:15s}: f* = {f_opt:10.6f}, "
                      f"误差 = {error:8.6f}, 迭代 = {iterations:3d}")
                
            except Exception as e:
                print(f"{method_name:15s}: 失败 - {str(e)[:30]}")
        
        # 找出最佳方法
        if results:
            best_accuracy = min(results, key=lambda x: x['error'])
            best_speed = min(results, key=lambda x: x['iterations'])
            
            print(f"\n最佳精度: {best_accuracy['method']} (误差: {best_accuracy['error']:.6f})")
            print(f"最快收敛: {best_speed['method']} (迭代: {best_speed['iterations']}次)")

def visualize_optimization_path():
    """可视化优化路径"""
    print("\n" + "=" * 60)
    print("优化路径可视化示例")
    print("=" * 60)
    
    # 创建优化器
    optimizer = TwoDimensionalOptimization(tolerance=1e-6, max_iterations=50)
    
    # 定义一个有趣的函数进行可视化
    def himmelblau(x):
        """Himmelblau函数，有四个全局最小值点"""
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    
    # 初始点
    x0 = np.array([-2.0, 3.0])
    
    # 使用几种不同的方法
    methods_to_visualize = [
        ("最速下降法", optimizer.steepest_descent),
        ("共轭梯度法", optimizer.conjugate_gradient),
        ("拟牛顿法(DFP)", optimizer.dfp_method)
    ]
    
    # 创建等高线图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 准备等高线数据
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = himmelblau(np.array([X[i, j], Y[i, j]]))
    
    for idx, (method_name, method) in enumerate(methods_to_visualize):
        try:
            x_opt, iterations, path = method(himmelblau, x0)
            
            ax = axes[idx]
            
            # 绘制等高线
            contour = ax.contour(X, Y, Z, levels=30, alpha=0.6)
            ax.clabel(contour, inline=True, fontsize=8)
            
            # 绘制优化路径
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'ro-', 
                   markersize=4, linewidth=2, alpha=0.7)
            ax.plot(x0[0], x0[1], 'go', markersize=10, label='起始点')
            ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15, label='最优点')
            
            ax.set_title(f'{method_name}\n迭代次数: {iterations}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            
            print(f"{method_name}: 找到最优点 ({x_opt[0]:.4f}, {x_opt[1]:.4f}), "
                  f"函数值 = {himmelblau(x_opt):.6f}, 迭代次数 = {iterations}")
            
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'{method_name}\n执行失败', 
                          transform=axes[idx].transAxes, ha='center', va='center')
            print(f"{method_name}: 执行失败 - {str(e)}")
    
    plt.tight_layout()
    plt.suptitle('Himmelblau函数优化路径比较', fontsize=14, y=1.02)
    plt.show()

def parameter_sensitivity_analysis():
    """参数敏感性分析"""
    print("\n" + "=" * 60)
    print("参数敏感性分析")
    print("=" * 60)
    
    # 测试不同的容差设置
    tolerances = [1e-4, 1e-6, 1e-8]
    
    def test_function(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    x0 = np.array([0.0, 0.0])
    
    print("测试不同容差对收敛的影响:")
    print("目标函数: f(x,y) = (x-1)² + (y-2)²")
    print()
    
    for tol in tolerances:
        optimizer = TwoDimensionalOptimization(tolerance=tol, max_iterations=100)
        
        print(f"容差 = {tol:.0e}:")
        
        # 测试几种方法
        methods = [
            ("最速下降法", optimizer.steepest_descent),
            ("共轭梯度法", optimizer.conjugate_gradient),
            ("拟牛顿法(DFP)", optimizer.dfp_method)
        ]
        
        for method_name, method in methods:
            try:
                x_opt, iterations, _ = method(test_function, x0)
                error = np.linalg.norm(x_opt - np.array([1.0, 2.0]))
                print(f"  {method_name:15s}: 误差 = {error:.8f}, 迭代 = {iterations:2d}")
            except:
                print(f"  {method_name:15s}: 执行失败")
        print()

if __name__ == "__main__":
    # 运行所有示例
    single_method_example()
    compare_all_methods()
    
    print("\n" + "="*60)
    print("是否运行可视化示例？(需要显示图形) (y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        visualize_optimization_path()
    
    print("\n" + "="*60)
    print("是否运行参数敏感性分析？(y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        parameter_sensitivity_analysis()
    
    print("\n程序结束！")
    print("您可以修改 usage_example.py 中的函数来测试您自己的优化问题。") 
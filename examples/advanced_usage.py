#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级用法示例
演示参数调节、性能对比、算法组合等高级功能
"""

import sys
import os
# 添加上级目录到Python路径，以便导入优化模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import time
from one_dim import OneDimensionalOptimization
from two_dim import TwoDimensionalOptimization
from cons_optimiz import ConstrainedOptimization

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def parameter_sensitivity_study():
    """参数敏感性研究"""
    print("=" * 70)
    print("参数敏感性研究")
    print("=" * 70)
    
    # 测试函数
    def test_func(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2
    
    x0 = np.array([0.0, 0.0])
    true_optimum = np.array([2.0, 1.0])
    
    # 测试不同的容差设置
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    
    print("容差对收敛的影响:")
    print("-" * 50)
    
    for tol in tolerances:
        optimizer = TwoDimensionalOptimization(tolerance=tol, max_iterations=100)
        
        start_time = time.time()
        x_opt, iterations, _ = optimizer.conjugate_gradient(test_func, x0)
        end_time = time.time()
        
        error = np.linalg.norm(x_opt - true_optimum)
        runtime = end_time - start_time
        
        print(f"容差 {tol:.0e}: 误差 = {error:.8f}, 迭代 = {iterations:2d}, "
              f"时间 = {runtime:.4f}s")

def algorithm_comparison_study():
    """算法性能对比研究"""
    print("\n" + "=" * 70)
    print("算法性能对比研究")
    print("=" * 70)
    
    # 定义多个测试函数
    test_functions = {
        '二次函数': (lambda x: (x[0]-1)**2 + (x[1]-2)**2, np.array([1.0, 2.0])),
        'Rosenbrock': (lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2, np.array([1.0, 1.0])),
        '椭圆函数': (lambda x: 2*x[0]**2 + x[1]**2, np.array([0.0, 0.0])),
        '双峰函数': (lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2, np.array([3.0, 2.0]))
    }
    
    optimizer = TwoDimensionalOptimization(tolerance=1e-6, max_iterations=200)
    
    methods = [
        ('最速下降法', optimizer.steepest_descent),
        ('共轭梯度法', optimizer.conjugate_gradient),
        ('拟牛顿法', optimizer.dfp_method),
        ('阻尼牛顿法', optimizer.damped_newton_method)
    ]
    
    results = {}
    
    for func_name, (func, true_opt) in test_functions.items():
        print(f"\n测试函数: {func_name}")
        print("-" * 40)
        
        x0 = np.array([-1.0, 1.0])  # 统一初始点
        func_results = []
        
        for method_name, method in methods:
            try:
                start_time = time.time()
                x_opt, iterations, path = method(func, x0)
                end_time = time.time()
                
                error = np.linalg.norm(x_opt - true_opt)
                runtime = end_time - start_time
                final_value = func(x_opt)
                
                func_results.append({
                    'method': method_name,
                    'error': error,
                    'iterations': iterations,
                    'runtime': runtime,
                    'final_value': final_value,
                    'path_length': len(path)
                })
                
                print(f"{method_name:15s}: 误差={error:.6f}, 迭代={iterations:3d}, "
                      f"时间={runtime:.4f}s, f*={final_value:.6f}")
                
            except Exception as e:
                print(f"{method_name:15s}: 失败 - {str(e)[:30]}")
        
        results[func_name] = func_results
    
    return results

def robust_optimization_study():
    """鲁棒性优化研究"""
    print("\n" + "=" * 70)
    print("鲁棒性优化研究（不同初始点）")
    print("=" * 70)
    
    def rosenbrock(x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    optimizer = TwoDimensionalOptimization(tolerance=1e-6, max_iterations=100)
    true_optimum = np.array([1.0, 1.0])
    
    # 多个不同的初始点
    initial_points = [
        np.array([-2.0, 2.0]),
        np.array([0.0, 0.0]),
        np.array([2.0, -1.0]),
        np.array([-1.0, -1.0]),
        np.array([1.5, 3.0])
    ]
    
    methods = [
        ('共轭梯度法', optimizer.conjugate_gradient),
        ('拟牛顿法', optimizer.dfp_method),
        ('阻尼牛顿法', optimizer.damped_newton_method)
    ]
    
    print("初始点对收敛的影响 (Rosenbrock函数):")
    print("-" * 60)
    
    for i, x0 in enumerate(initial_points):
        print(f"\n初始点 {i+1}: ({x0[0]:5.1f}, {x0[1]:5.1f})")
        
        for method_name, method in methods:
            try:
                x_opt, iterations, _ = method(rosenbrock, x0)
                error = np.linalg.norm(x_opt - true_optimum)
                success = "成功" if error < 1e-3 else "失败"
                
                print(f"  {method_name:15s}: 误差={error:.6f}, 迭代={iterations:3d}, {success}")
                
            except Exception:
                print(f"  {method_name:15s}: 算法发散")

def constraint_handling_comparison():
    """约束处理方法对比"""
    print("\n" + "=" * 70)
    print("约束处理方法对比")
    print("=" * 70)
    
    # 定义约束优化问题
    def objective(x):
        return (x[0] - 3)**2 + (x[1] - 2)**2
    
    def eq_constraint(x):
        return x[0]**2 + x[1]**2 - 5  # x² + y² = 5
    
    def ineq_constraint(x):
        return x[0] + x[1] - 1  # x + y >= 1
    
    optimizer = ConstrainedOptimization(tolerance=1e-6, max_iterations=50)
    x0 = np.array([2.0, 1.0])
    
    print("问题: min (x-3)² + (y-2)², s.t. x²+y²=5, x+y≥1")
    print(f"初始点: ({x0[0]}, {x0[1]})")
    print("-" * 50)
    
    methods = [
        ('约束坐标轮换法', lambda: optimizer.constrained_coordinate_descent(
            objective, x0, [eq_constraint], [ineq_constraint])),
        ('拉格朗日法', lambda: optimizer.lagrange_method(
            objective, x0, [eq_constraint])),
        ('惩罚函数法', lambda: optimizer.penalty_method(
            objective, x0, [eq_constraint], [ineq_constraint]))
    ]
    
    for method_name, method_func in methods:
        try:
            start_time = time.time()
            x_opt, iterations, path = method_func()
            end_time = time.time()
            
            runtime = end_time - start_time
            final_value = objective(x_opt)
            is_feasible = optimizer.is_feasible(x_opt, [eq_constraint], [ineq_constraint])
            
            print(f"{method_name:20s}: f*={final_value:.6f}, 迭代={iterations:3d}, "
                  f"时间={runtime:.4f}s, 可行={'是' if is_feasible else '否'}")
            
        except Exception as e:
            print(f"{method_name:20s}: 失败 - {str(e)[:30]}")

def adaptive_parameter_optimization():
    """自适应参数优化"""
    print("\n" + "=" * 70)
    print("自适应参数优化示例")
    print("=" * 70)
    
    def difficult_function(x):
        """难优化的函数（条件数很大）"""
        return 100 * x[0]**2 + x[1]**2
    
    x0 = np.array([1.0, 10.0])
    true_optimum = np.array([0.0, 0.0])
    
    print("难优化函数: f(x,y) = 100x² + y² (条件数=100)")
    print("比较标准参数和调整参数的效果:")
    print("-" * 50)
    
    # 标准参数
    optimizer1 = TwoDimensionalOptimization(tolerance=1e-6, max_iterations=200)
    start_time = time.time()
    x_opt1, iter1, path1 = optimizer1.steepest_descent(difficult_function, x0)
    time1 = time.time() - start_time
    error1 = np.linalg.norm(x_opt1 - true_optimum)
    
    # 调整参数（更严格的容差，更多迭代）
    optimizer2 = TwoDimensionalOptimization(tolerance=1e-8, max_iterations=500)
    start_time = time.time()
    x_opt2, iter2, path2 = optimizer2.conjugate_gradient(difficult_function, x0)
    time2 = time.time() - start_time
    error2 = np.linalg.norm(x_opt2 - true_optimum)
    
    print(f"最速下降法(标准): 误差={error1:.8f}, 迭代={iter1:3d}, 时间={time1:.4f}s")
    print(f"共轭梯度法(调整): 误差={error2:.8f}, 迭代={iter2:3d}, 时间={time2:.4f}s")
    
    # 可视化比较
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    path1_array = np.array(path1)
    plt.plot(path1_array[:, 0], path1_array[:, 1], 'ro-', markersize=3, label='最速下降法')
    plt.plot(0, 0, 'g*', markersize=15, label='真实最优点')
    plt.title('最速下降法路径')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    path2_array = np.array(path2)
    plt.plot(path2_array[:, 0], path2_array[:, 1], 'bo-', markersize=3, label='共轭梯度法')
    plt.plot(0, 0, 'g*', markersize=15, label='真实最优点')
    plt.title('共轭梯度法路径')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def hybrid_optimization_approach():
    """混合优化方法"""
    print("\n" + "=" * 70)
    print("混合优化方法示例")
    print("=" * 70)
    
    def complex_function(x):
        """复杂的多峰函数"""
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2 + 0.1*np.sin(10*x[0])*np.sin(10*x[1])
    
    x0 = np.array([0.0, 0.0])
    
    print("混合策略: 先用全局搜索，再用局部优化")
    print("-" * 50)
    
    optimizer = TwoDimensionalOptimization(tolerance=1e-6, max_iterations=50)
    
    # 第一阶段：粗略搜索（最速下降法）
    print("第一阶段：全局粗略搜索")
    x_coarse, iter1, path1 = optimizer.steepest_descent(complex_function, x0)
    print(f"粗略搜索结果: x=({x_coarse[0]:.4f}, {x_coarse[1]:.4f}), "
          f"f={complex_function(x_coarse):.6f}, 迭代={iter1}")
    
    # 第二阶段：精细优化（牛顿法）
    print("\n第二阶段：局部精细优化")
    x_fine, iter2, path2 = optimizer.damped_newton_method(complex_function, x_coarse)
    print(f"精细优化结果: x=({x_fine[0]:.4f}, {x_fine[1]:.4f}), "
          f"f={complex_function(x_fine):.6f}, 迭代={iter2}")
    
    # 与单一方法比较
    print("\n单一方法对比:")
    x_single, iter_single, _ = optimizer.conjugate_gradient(complex_function, x0)
    print(f"共轭梯度法   : x=({x_single[0]:.4f}, {x_single[1]:.4f}), "
          f"f={complex_function(x_single):.6f}, 迭代={iter_single}")
    
    total_iterations = iter1 + iter2
    print(f"\n混合方法总迭代: {total_iterations}, 单一方法: {iter_single}")
    improvement = complex_function(x_single) - complex_function(x_fine)
    print(f"函数值改进: {improvement:.6f}")

def performance_profiling():
    """性能剖析"""
    print("\n" + "=" * 70)
    print("性能剖析")
    print("=" * 70)
    
    def benchmark_function(x):
        """基准测试函数"""
        return (x[0] - 1)**2 + (x[1] - 1)**2
    
    x0 = np.array([0.0, 0.0])
    n_runs = 10  # 运行次数
    
    optimizer = TwoDimensionalOptimization(tolerance=1e-6, max_iterations=100)
    methods = [
        ('最速下降法', optimizer.steepest_descent),
        ('共轭梯度法', optimizer.conjugate_gradient),
        ('拟牛顿法', optimizer.dfp_method)
    ]
    
    print(f"基准测试 (平均 {n_runs} 次运行):")
    print("-" * 40)
    
    for method_name, method in methods:
        times = []
        iterations = []
        
        for _ in range(n_runs):
            start_time = time.time()
            _, iter_count, _ = method(benchmark_function, x0)
            end_time = time.time()
            
            times.append(end_time - start_time)
            iterations.append(iter_count)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_iter = np.mean(iterations)
        
        print(f"{method_name:15s}: 平均时间={avg_time:.6f}±{std_time:.6f}s, "
              f"平均迭代={avg_iter:.1f}")

if __name__ == "__main__":
    print("🚀 高级用法示例演示")
    print("=" * 70)
    
    # 运行各种高级功能演示
    parameter_sensitivity_study()
    algorithm_comparison_study()
    robust_optimization_study()
    constraint_handling_comparison()
    
    print("\n" + "="*70)
    print("是否运行自适应参数优化和混合方法演示？(需要显示图形) (y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        adaptive_parameter_optimization()
        hybrid_optimization_approach()
    
    print("\n" + "="*70)
    print("是否运行性能剖析？(y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        performance_profiling()
    
    print("\n程序结束！")
    print("这些高级功能可以帮助您:")
    print("1. 选择合适的算法参数")
    print("2. 对比不同算法的性能")
    print("3. 处理复杂的优化问题")
    print("4. 组合多种方法提高效果") 
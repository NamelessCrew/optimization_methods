#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义函数示例
演示如何定义和测试您自己的优化函数
"""

import sys
import os
# 添加上级目录到Python路径，以便导入优化模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from one_dim import OneDimensionalOptimization
from two_dim import TwoDimensionalOptimization
from cons_optimiz import ConstrainedOptimization

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===============================
# 一维自定义函数示例
# ===============================

def sinc_function(x):
    """sinc函数: sin(x)/x"""
    if abs(x) < 1e-10:
        return 1.0
    return np.sin(x) / x

def polynomial_function(x):
    """多项式函数: x^4 - 8x^3 + 18x^2 - 16x + 5"""
    return x**4 - 8*x**3 + 18*x**2 - 16*x + 5

def exponential_function(x):
    """指数函数: e^(-x) + x^2"""
    return np.exp(-x) + x**2

# ===============================
# 二维自定义函数示例  
# ===============================

def ackley_function(x):
    """Ackley函数（多峰函数）"""
    a, b, c = 20, 0.2, 2*np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    
    return term1 + term2 + a + np.exp(1)

def beale_function(x):
    """Beale函数"""
    return ((1.5 - x[0] + x[0]*x[1])**2 + 
            (2.25 - x[0] + x[0]*x[1]**2)**2 + 
            (2.625 - x[0] + x[0]*x[1]**3)**2)

def booth_function(x):
    """Booth函数"""
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def matyas_function(x):
    """Matyas函数"""
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

# ===============================
# 约束优化自定义函数示例
# ===============================

def engineering_objective(x):
    """工程设计目标函数：最小化材料成本"""
    # x[0]: 宽度, x[1]: 高度
    return x[0] * x[1] * 10  # 成本系数为10

def strength_constraint(x):
    """强度约束：横截面积必须大于某个值"""
    return x[0] * x[1] - 50  # 面积 >= 50

def size_constraint1(x):
    """尺寸约束1：宽度限制"""
    return x[0] - 2  # 宽度 >= 2

def size_constraint2(x):
    """尺寸约束2：高度限制"""
    return x[1] - 3  # 高度 >= 3

def ratio_constraint(x):
    """比例约束：宽高比限制"""
    return 5 - x[0]/x[1]  # 宽高比 <= 5

# ===============================
# 测试函数
# ===============================

def test_one_dim_custom():
    """测试一维自定义函数"""
    print("=" * 60)
    print("一维自定义函数测试")
    print("=" * 60)
    
    optimizer = OneDimensionalOptimization(tolerance=1e-6)
    
    functions = [
        (sinc_function, "sinc函数", -10, 10),
        (polynomial_function, "多项式函数", 0, 5),
        (exponential_function, "指数函数", -2, 3)
    ]
    
    for func, name, a, b in functions:
        print(f"\n测试函数: {name}")
        print(f"搜索区间: [{a}, {b}]")
        
        try:
            # 使用黄金分割法
            x_opt, iterations = optimizer.golden_section_method(func, a, b)
            print(f"黄金分割法: x* = {x_opt:.6f}, f(x*) = {func(x_opt):.6f}, 迭代 = {iterations}")
            
            # 使用牛顿法（从中点开始）
            x_opt, iterations = optimizer.newton_method(func, (a+b)/2)
            print(f"牛顿法    : x* = {x_opt:.6f}, f(x*) = {func(x_opt):.6f}, 迭代 = {iterations}")
            
        except Exception as e:
            print(f"测试失败: {str(e)}")

def test_two_dim_custom():
    """测试二维自定义函数"""
    print("\n" + "=" * 60)
    print("二维自定义函数测试")
    print("=" * 60)
    
    optimizer = TwoDimensionalOptimization(tolerance=1e-6)
    
    functions = [
        (ackley_function, "Ackley函数", np.array([1.0, 1.0])),
        (beale_function, "Beale函数", np.array([1.0, 1.0])),
        (booth_function, "Booth函数", np.array([0.0, 0.0])),
        (matyas_function, "Matyas函数", np.array([1.0, 1.0]))
    ]
    
    for func, name, x0 in functions:
        print(f"\n测试函数: {name}")
        print(f"初始点: ({x0[0]}, {x0[1]})")
        
        try:
            # 使用共轭梯度法
            x_opt, iterations, _ = optimizer.conjugate_gradient(func, x0)
            print(f"共轭梯度法: x* = ({x_opt[0]:.4f}, {x_opt[1]:.4f}), "
                  f"f(x*) = {func(x_opt):.6f}, 迭代 = {iterations}")
            
            # 使用拟牛顿法
            x_opt, iterations, _ = optimizer.dfp_method(func, x0)
            print(f"拟牛顿法  : x* = ({x_opt[0]:.4f}, {x_opt[1]:.4f}), "
                  f"f(x*) = {func(x_opt):.6f}, 迭代 = {iterations}")
            
        except Exception as e:
            print(f"测试失败: {str(e)}")

def test_constrained_custom():
    """测试约束优化自定义问题"""
    print("\n" + "=" * 60)
    print("约束优化自定义问题测试")
    print("=" * 60)
    
    optimizer = ConstrainedOptimization(tolerance=1e-6)
    
    print("工程设计问题：最小化材料成本")
    print("目标函数: min 10*width*height")
    print("约束条件: width*height >= 50, width >= 2, height >= 3, width/height <= 5")
    
    x0 = np.array([5.0, 10.0])
    eq_constraints = []
    ineq_constraints = [strength_constraint, size_constraint1, size_constraint2, ratio_constraint]
    
    try:
        # 使用惩罚函数法
        x_opt, iterations, _ = optimizer.penalty_method(
            engineering_objective, x0, eq_constraints, ineq_constraints)
        
        print(f"\n惩罚函数法结果:")
        print(f"最优解: width = {x_opt[0]:.4f}, height = {x_opt[1]:.4f}")
        print(f"最小成本: {engineering_objective(x_opt):.4f}")
        print(f"迭代次数: {iterations}")
        
        # 验证约束
        print(f"\n约束验证:")
        print(f"面积约束 (>=50): {x_opt[0]*x_opt[1]:.2f}")
        print(f"宽度约束 (>=2): {x_opt[0]:.2f}")
        print(f"高度约束 (>=3): {x_opt[1]:.2f}")
        print(f"比例约束 (<=5): {x_opt[0]/x_opt[1]:.2f}")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")

def visualize_custom_function():
    """可视化自定义函数"""
    print("\n" + "=" * 60)
    print("自定义函数可视化")
    print("=" * 60)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 一维sinc函数
    ax1 = axes[0, 0]
    x = np.linspace(-10, 10, 1000)
    y = [sinc_function(xi) for xi in x]
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.set_title('sinc函数: sin(x)/x')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Ackley函数（二维等高线）
    ax2 = axes[0, 1]
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = ackley_function(np.array([X[i, j], Y[i, j]]))
    
    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_title('Ackley函数等高线')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # 3. Beale函数
    ax3 = axes[1, 0]
    x_range = np.linspace(-4.5, 4.5, 100)
    y_range = np.linspace(-4.5, 4.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = beale_function(np.array([X[i, j], Y[i, j]]))
    
    contour = ax3.contour(X, Y, Z, levels=20)
    ax3.clabel(contour, inline=True, fontsize=8)
    ax3.set_title('Beale函数等高线')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    
    # 4. 约束优化问题可行域
    ax4 = axes[1, 1]
    x_vals = np.linspace(0, 15, 100)
    y_vals = np.linspace(0, 15, 100)
    
    # 绘制约束边界
    y_strength = 50 / x_vals  # 强度约束边界
    mask = (x_vals >= 2) & (y_strength >= 3) & (x_vals/y_strength <= 5)
    
    ax4.plot(x_vals[mask], y_strength[mask], 'r-', linewidth=2, label='强度约束')
    ax4.axvline(x=2, color='g', linestyle='--', label='宽度约束')
    ax4.axhline(y=3, color='b', linestyle='--', label='高度约束')
    
    # 比例约束
    y_ratio = x_vals / 5
    ax4.plot(x_vals, y_ratio, 'orange', linestyle='--', label='比例约束')
    
    ax4.set_xlim(0, 15)
    ax4.set_ylim(0, 15)
    ax4.set_title('工程设计问题约束')
    ax4.set_xlabel('宽度')
    ax4.set_ylabel('高度')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行所有测试
    test_one_dim_custom()
    test_two_dim_custom()
    test_constrained_custom()
    
    print("\n" + "="*60)
    print("是否显示可视化图形？(y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        visualize_custom_function()
    
    print("\n程序结束！")
    print("您可以修改本文件来添加和测试您自己的优化函数。") 
import numpy as np
from scipy.misc import derivative

def is_convex(f, a, b, n_points=1000, h=1e-5):
    # 在区间内均匀采样点
    x = np.linspace(a, b, n_points)
    
    # 计算每个点的二阶导数
    second_derivatives = []
    for x0 in x:
        # 使用数值方法计算二阶导数
        second_der = derivative(lambda x: derivative(f, x, dx=h), x0, dx=h)
        second_derivatives.append(second_der)
    
    # 检查所有二阶导数是否都非负
    return all(d2 >= -1e-10 for d2 in second_derivatives)

if __name__ == "__main__":
    def f1(x):
        return 0.5*x**3+2*x**2+2*x+1
    def f2(x):
        return -0.5*x**3-2*x**2-2*x-1
    
    # 测试结果
    print("f(x) = 0.5*x^3+2*x^2+2*x+1 是否为凸函数:", is_convex(f1, -10, 10))
    print("f(x) = -0.5*x**3-2*x**2-2*x-1 是否为凸函数:", is_convex(f2, -10, 10))

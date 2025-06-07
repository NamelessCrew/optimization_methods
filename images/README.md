# 效果图存储目录

此目录用于存放项目的效果图和演示图片。

## 需要添加的图片

请在此目录下添加以下效果图：

### 📊 一维优化效果图
- **文件名**: `one_dim_demo.png`
- **内容**: 六种一维优化算法在不同测试函数上的表现比较
- **建议尺寸**: 800x600 像素
- **获取方式**: 运行 `python one_dim.py` 并保存生成的图表

### 📈 二维优化路径追踪
- **文件名**: `two_dim_demo.png`
- **内容**: 七种二维优化算法的收敛路径可视化
- **建议尺寸**: 800x600 像素
- **获取方式**: 运行 `python two_dim.py` 或 `python usage_example.py` 并保存路径图

### ⚖️ 约束优化可视化
- **文件名**: `constrained_demo.png`
- **内容**: 三种约束优化方法处理约束问题的效果
- **建议尺寸**: 800x600 像素
- **获取方式**: 运行 `python cons_optimiz.py` 并保存约束优化可视化图

### 📊 性能对比分析
- **文件名**: `performance.png`
- **内容**: 不同算法在收敛速度、精度和稳定性方面的综合比较
- **建议尺寸**: 800x600 像素
- **获取方式**: 运行 `python examples/advanced_usage.py` 并保存性能对比图

## 图片要求

- **格式**: PNG (推荐) 或 JPG
- **质量**: 高清晰度，适合在 README 中展示
- **内容**: 图表清晰，文字可读，颜色搭配合理
- **语言**: 支持中文显示的图表

## 添加步骤

1. 运行相应的 Python 文件生成图表
2. 使用 matplotlib 的 `plt.savefig()` 保存图片到此目录
3. 确保文件名与 README.md 中引用的名称一致
4. 可选：使用图片压缩工具优化文件大小

## 示例代码

```python
import matplotlib.pyplot as plt

# 生成图表后
plt.savefig('images/your_image_name.png', dpi=300, bbox_inches='tight')
plt.show()
```

添加图片后，GitHub README 将能正确显示这些效果图。 
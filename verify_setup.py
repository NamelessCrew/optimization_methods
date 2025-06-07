#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目设置验证脚本
检查所有文件是否正确设置，依赖是否满足
"""

import os
import sys
import importlib.util

def check_file_exists(filename, description):
    """检查文件是否存在"""
    if os.path.exists(filename):
        print(f"✅ {description}: {filename}")
        return True
    else:
        print(f"❌ {description}: {filename} - 文件不存在")
        return False

def check_directory_exists(dirname, description):
    """检查目录是否存在"""
    if os.path.isdir(dirname):
        print(f"✅ {description}: {dirname}/")
        return True
    else:
        print(f"❌ {description}: {dirname}/ - 目录不存在")
        return False

def check_python_module(module_name):
    """检查Python模块是否可导入"""
    try:
        __import__(module_name)
        print(f"✅ Python模块: {module_name}")
        return True
    except ImportError:
        print(f"❌ Python模块: {module_name} - 导入失败")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查依赖包:")
    print("-" * 40)
    
    dependencies = ['numpy', 'scipy', 'matplotlib']
    all_good = True
    
    for dep in dependencies:
        if not check_python_module(dep):
            all_good = False
    
    return all_good

def check_core_files():
    """检查核心文件"""
    print("\n📄 检查核心文件:")
    print("-" * 40)
    
    files = [
        ('README.md', 'README文档'),
        ('LICENSE', 'MIT许可证'),
        ('requirements.txt', '依赖列表'),
        ('.gitignore', 'Git忽略文件'),
        ('one_dim.py', '一维优化模块'),
        ('two_dim.py', '二维优化模块'),
        ('cons_optimiz.py', '约束优化模块'),
        ('usage_example.py', '使用示例'),
        ('fuc.py', '辅助函数')
    ]
    
    all_good = True
    for filename, description in files:
        if not check_file_exists(filename, description):
            all_good = False
    
    return all_good

def check_example_files():
    """检查示例文件"""
    print("\n📝 检查示例文件:")
    print("-" * 40)
    
    files = [
        ('examples/custom_functions.py', '自定义函数示例'),
        ('examples/advanced_usage.py', '高级用法示例')
    ]
    
    all_good = True
    for filename, description in files:
        if not check_file_exists(filename, description):
            all_good = False
    
    return all_good

def check_directories():
    """检查目录结构"""
    print("\n📁 检查目录结构:")
    print("-" * 40)
    
    directories = [
        ('images', '效果图目录'),
        ('examples', '示例目录')
    ]
    
    all_good = True
    for dirname, description in directories:
        if not check_directory_exists(dirname, description):
            all_good = False
    
    return all_good

def check_imports():
    """检查模块导入"""
    print("\n🔧 检查模块导入:")
    print("-" * 40)
    
    modules = [
        ('one_dim', '一维优化模块'),
        ('two_dim', '二维优化模块'),
        ('cons_optimiz', '约束优化模块')
    ]
    
    all_good = True
    for module_name, description in modules:
        try:
            spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
            if spec is None:
                print(f"❌ {description}: 无法找到模块文件")
                all_good = False
                continue
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ {description}: 导入成功")
        except Exception as e:
            print(f"❌ {description}: 导入失败 - {str(e)}")
            all_good = False
    
    return all_good

def print_project_stats():
    """打印项目统计信息"""
    print("\n📊 项目统计:")
    print("-" * 40)
    
    # 统计Python文件
    py_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    
    # 统计代码行数
    total_lines = 0
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_lines += lines
                print(f"  {py_file}: {lines} 行")
        except:
            print(f"  {py_file}: 无法读取")
    
    print(f"\n📈 总计:")
    print(f"  Python文件: {len(py_files)} 个")
    print(f"  代码总行数: {total_lines} 行")

def check_git_setup():
    """检查Git设置"""
    print("\n🔄 检查Git设置:")
    print("-" * 40)
    
    if os.path.exists('.git'):
        print("✅ Git仓库已初始化")
        
        # 检查是否有未跟踪的文件
        import subprocess
        try:
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                untracked = [line for line in result.stdout.split('\n') 
                           if line.startswith('??')]
                if untracked:
                    print(f"⚠️  有 {len(untracked)} 个未跟踪的文件")
                else:
                    print("✅ 所有文件已跟踪")
            else:
                print("⚠️  无法检查Git状态")
        except:
            print("⚠️  Git命令不可用")
    else:
        print("❌ Git仓库未初始化")
        print("   运行: git init")

def provide_setup_instructions():
    """提供设置说明"""
    print("\n🚀 准备发布到GitHub:")
    print("-" * 40)
    print("1. 确保所有依赖已安装:")
    print("   pip install -r requirements.txt")
    print()
    print("2. 生成效果图:")
    print("   python one_dim.py")
    print("   python two_dim.py") 
    print("   python cons_optimiz.py")
    print("   python examples/advanced_usage.py")
    print()
    print("3. 将图片保存到 images/ 目录")
    print()
    print("4. 初始化Git仓库 (如果还没有):")
    print("   git init")
    print("   git add .")
    print("   git commit -m 'Initial commit'")
    print()
    print("5. 连接到GitHub仓库:")
    print("   git remote add origin https://github.com/yourusername/optimization_methods.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print()
    print("6. 在README.md中更新:")
    print("   - 替换 'yourusername' 为您的GitHub用户名")
    print("   - 替换作者信息")
    print("   - 添加实际的效果图")

def main():
    """主函数"""
    print("🔍 优化方法项目验证")
    print("=" * 50)
    
    checks = [
        check_dependencies(),
        check_core_files(),
        check_directories(),
        check_example_files(),
        check_imports()
    ]
    
    print_project_stats()
    check_git_setup()
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 所有检查通过！项目已准备好发布到GitHub。")
    else:
        print("⚠️  发现一些问题，请修复后再发布。")
    
    provide_setup_instructions()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
测试CSV数据集和Ollama CoT生成器
"""

import os
import sys
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.csv_dataset import CSVBinbridgeDataset, OllamaCOTGenerator


def test_csv_dataset():
    """测试CSV数据集加载"""
    print("=" * 60)
    print("测试 CSVBinbridgeDataset")
    print("=" * 60)
    
    # 检查CSV文件是否存在
    csv_path = "func_pairs_with_strings_train_with_eval.csv"
    if not os.path.exists(csv_path):
        print(f"警告: CSV文件不存在: {csv_path}")
        print("请提供正确的CSV文件路径")
        return False
    
    # 读取CSV的前几行来测试
    print(f"\n读取CSV文件: {csv_path}")
    try:
        df = pd.read_csv(csv_path, nrows=5)
        print(f"CSV文件列: {list(df.columns)}")
        print(f"前5行数据:")
        print(df.head())
        
        # 检查必需的列
        required_cols = ['function_name', 'asm_func']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"\n错误: 缺少必需的列: {missing_cols}")
            return False
        
        print("\n✓ CSV文件格式正确")
        return True
    
    except Exception as e:
        print(f"\n错误: 无法读取CSV文件: {e}")
        return False


def test_ollama_connection():
    """测试Ollama连接"""
    print("\n" + "=" * 60)
    print("测试 Ollama 连接")
    print("=" * 60)
    
    try:
        generator = OllamaCOTGenerator(
            model_name="llama3",
            base_url="http://localhost:11434"
        )
        print("\n✓ Ollama连接成功")
        
        # 测试生成一个简单的分析
        print("\n测试生成CoT分析...")
        test_assembly = """push rbp
mov rbp, rsp
mov dword ptr [rbp-0x4], edi
mov eax, dword ptr [rbp-0x4]
add eax, 1
pop rbp
ret"""
        
        analysis = generator.generate_analysis(
            source_code=None,
            assembly_code=test_assembly,
            function_name="increment",
            arch="x86_64",
            opt_level="O2",
            max_tokens=200
        )
        
        if analysis:
            print(f"\n✓ 成功生成分析:")
            print(f"  {analysis[:200]}...")
            return True
        else:
            print("\n✗ 生成分析失败")
            return False
    
    except Exception as e:
        print(f"\n✗ Ollama连接失败: {e}")
        print("\n请确保:")
        print("1. Ollama服务正在运行 (ollama serve)")
        print("2. 模型已安装 (ollama pull llama3)")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("CSV数据集和Ollama CoT生成器测试")
    print("=" * 60)
    
    results = []
    
    # 测试CSV数据集
    results.append(("CSV数据集", test_csv_dataset()))
    
    # 测试Ollama连接（可选）
    test_ollama = input("\n是否测试Ollama连接? (y/n): ").strip().lower()
    if test_ollama == 'y':
        results.append(("Ollama连接", test_ollama_connection()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ 所有测试通过!")
    else:
        print("\n✗ 部分测试失败，请检查上述错误信息")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


#!/usr/bin/env python3
"""
从CSV文件生成CoT标注的脚本
使用本地Ollama模型生成思维链分析
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csv_dataset import OllamaCOTGenerator


def main():
    parser = argparse.ArgumentParser(
        description="使用Ollama从CSV文件生成CoT标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成前1000个样本的CoT标注
  python scripts/generate_cot_from_csv.py \\
      --csv_path func_pairs_with_strings_train_with_eval.csv \\
      --output cot_annotations.json \\
      --model llama3 \\
      --max_samples 1000

  # 从索引1000开始生成5000个样本
  python scripts/generate_cot_from_csv.py \\
      --csv_path func_pairs_with_strings_train_with_eval.csv \\
      --output cot_annotations_2.json \\
      --model llama3 \\
      --start_idx 1000 \\
      --end_idx 6000

  # 使用不同的Ollama模型和服务器
  python scripts/generate_cot_from_csv.py \\
      --csv_path func_pairs_with_strings_train_with_eval.csv \\
      --output cot_annotations.json \\
      --model qwen2.5:7b \\
      --base_url http://192.168.1.100:11434
        """
    )
    
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="输入CSV文件路径"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出JSON文件路径"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Ollama模型名称 (默认: llama3)"
    )
    
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API基础URL (默认: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        help="最大处理样本数"
    )
    
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="起始索引 (默认: 0)"
    )
    
    parser.add_argument(
        "--end_idx",
        type=int,
        help="结束索引 (不包含)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度 (默认: 0.7)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="最大生成token数 (默认: 500)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="请求超时时间（秒）(默认: 300)"
    )
    
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="跳过已有标注的样本 (默认: True)"
    )
    
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="不跳过已有标注的样本（重新生成）"
    )
    
    args = parser.parse_args()
    
    # 检查CSV文件是否存在
    if not os.path.exists(args.csv_path):
        print(f"错误: CSV文件不存在: {args.csv_path}")
        sys.exit(1)
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Ollama CoT 生成器")
    print("=" * 60)
    print(f"CSV文件: {args.csv_path}")
    print(f"输出文件: {args.output}")
    print(f"模型: {args.model}")
    print(f"Ollama URL: {args.base_url}")
    print(f"起始索引: {args.start_idx}")
    if args.end_idx:
        print(f"结束索引: {args.end_idx}")
    if args.max_samples:
        print(f"最大样本数: {args.max_samples}")
    print(f"温度: {args.temperature}")
    print(f"最大tokens: {args.max_tokens}")
    print(f"超时: {args.timeout}秒")
    print("=" * 60)
    
    # 初始化生成器
    try:
        generator = OllamaCOTGenerator(
            model_name=args.model,
            base_url=args.base_url,
            timeout=args.timeout
        )
    except Exception as e:
        print(f"错误: 无法初始化Ollama生成器: {e}")
        print("\n请确保:")
        print("1. Ollama服务正在运行")
        print(f"2. 模型 '{args.model}' 已安装 (运行: ollama pull {args.model})")
        print(f"3. Ollama API可访问: {args.base_url}")
        sys.exit(1)
    
    # 生成CoT标注
    try:
        generator.batch_generate_from_csv(
            csv_path=args.csv_path,
            output_path=args.output,
            max_samples=args.max_samples,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            skip_existing=args.skip_existing,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        print("\n✓ CoT标注生成完成!")
    except KeyboardInterrupt:
        print("\n\n用户中断，已保存部分结果")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


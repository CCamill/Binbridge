#!/usr/bin/env python3
"""
使用 Hugging Face 下载模型（支持镜像和断点续传）
作为 Ollama 网络问题的替代方案
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_model_with_mirror(model_name: str, use_mirror: bool = True):
    """
    使用 Hugging Face 下载模型，支持镜像源
    
    Args:
        model_name: 模型名称，如 "Qwen/Qwen2.5-Coder-30B-Instruct"
        use_mirror: 是否使用国内镜像（hf-mirror.com）
    """
    from huggingface_hub import snapshot_download
    
    # 如果使用镜像，设置环境变量
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"使用镜像源: https://hf-mirror.com")
    
    print(f"开始下载模型: {model_name}")
    print("这可能需要较长时间，请耐心等待...")
    print("支持断点续传，如果中断可以重新运行此脚本")
    
    try:
        # 下载到本地缓存
        cache_dir = snapshot_download(
            repo_id=model_name,
            resume_download=True,  # 支持断点续传
            local_files_only=False
        )
        
        print(f"\n✓ 模型下载完成!")
        print(f"模型路径: {cache_dir}")
        return cache_dir
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n提示:")
        print("1. 检查网络连接")
        print("2. 如果使用镜像，尝试不使用镜像: use_mirror=False")
        print("3. 检查模型名称是否正确")
        print("4. 确保有足够的磁盘空间")
        raise


def download_with_transformers(model_name: str, use_mirror: bool = True):
    """
    使用 transformers 库下载模型（会同时下载分词器）
    """
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"使用镜像源: https://hf-mirror.com")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"开始下载模型和分词器: {model_name}")
    
    try:
        # 下载分词器
        print("\n下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ 分词器下载完成")
        
        # 注意：这里不实际加载模型权重，只是下载配置
        # 实际加载模型时才会下载权重
        print("\n模型配置已准备就绪")
        print("实际模型权重将在首次加载时下载")
        
        return tokenizer
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="使用 Hugging Face 下载模型（支持镜像）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-30B-Instruct",
        help="模型名称 (默认: Qwen/Qwen2.5-Coder-30B-Instruct)"
    )
    parser.add_argument(
        "--use_mirror",
        action="store_true",
        default=True,
        help="使用国内镜像源 (默认: True)"
    )
    parser.add_argument(
        "--no_mirror",
        dest="use_mirror",
        action="store_false",
        help="不使用镜像源"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["snapshot", "transformers"],
        default="snapshot",
        help="下载方法: snapshot (完整下载) 或 transformers (按需下载)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hugging Face 模型下载工具")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"方法: {args.method}")
    print(f"镜像: {'是' if args.use_mirror else '否'}")
    print("=" * 60)
    
    try:
        if args.method == "snapshot":
            download_model_with_mirror(args.model, args.use_mirror)
        else:
            download_with_transformers(args.model, args.use_mirror)
        
        print("\n" + "=" * 60)
        print("下载完成!")
        print("=" * 60)
        print("\n使用方式:")
        print("1. 在代码中直接使用模型名称，会自动使用缓存")
        print("2. 或者指定本地路径")
        print("\n示例:")
        print(f'  model = AutoModelForCausalLM.from_pretrained("{args.model}")')
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
        print("可以重新运行此脚本继续下载（支持断点续传）")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()




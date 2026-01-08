#!/usr/bin/env python3
"""
测试 CodeLlama 34B 4bit 量化版本的显存占用
"""

import torch
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


def format_bytes(bytes_value):
    """格式化字节数为可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_gpu_memory_info():
    """获取GPU显存信息"""
    if not torch.cuda.is_available():
        return None
    
    info = {}
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        total = torch.cuda.get_device_properties(i).total_memory
        
        info[f"GPU {i}"] = {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "free": total - reserved
        }
    
    return info


def print_memory_info(stage="", gpu_info=None):
    """打印显存信息"""
    print("\n" + "=" * 60)
    if stage:
        print(f"显存状态: {stage}")
    print("=" * 60)
    
    if gpu_info is None:
        gpu_info = get_gpu_memory_info()
    
    if gpu_info is None:
        print("未检测到GPU")
        return
    
    for gpu_name, mem_info in gpu_info.items():
        print(f"\n{gpu_name}:")
        print(f"  总显存:     {format_bytes(mem_info['total'])}")
        print(f"  已分配:     {format_bytes(mem_info['allocated'])}")
        print(f"  已保留:     {format_bytes(mem_info['reserved'])}")
        print(f"  可用显存:   {format_bytes(mem_info['free'])}")
        print(f"  使用率:     {mem_info['reserved'] / mem_info['total'] * 100:.2f}%")


def test_codellama_34b_4bit():
    """测试 CodeLlama 34B 4bit 量化版本的显存占用"""
    
    # 模型名称 - 根据实际情况调整
    model_name = "codellama/CodeLlama-34b-Instruct-hf"
    
    print("=" * 60)
    print("CodeLlama 34B 4bit 量化显存测试")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"量化: 4bit NF4")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 60)
    
    # 记录初始显存
    initial_memory = get_gpu_memory_info()
    print_memory_info("初始状态", initial_memory)
    
    try:
        # 配置 4bit 量化
        print("\n配置 4bit NF4 量化...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # 加载分词器
        print(f"\n加载分词器: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 记录加载分词器后的显存
        after_tokenizer_memory = get_gpu_memory_info()
        print_memory_info("加载分词器后", after_tokenizer_memory)
        
        # 加载模型
        print(f"\n加载模型 (4bit量化)...")
        print("这可能需要几分钟时间，请耐心等待...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # 记录加载模型后的显存
        after_model_memory = get_gpu_memory_info()
        print_memory_info("加载模型后", after_model_memory)
        
        # 计算显存增量
        if initial_memory and after_model_memory:
            print("\n" + "=" * 60)
            print("显存占用分析")
            print("=" * 60)
            for gpu_name in initial_memory.keys():
                if gpu_name in after_model_memory:
                    initial_allocated = initial_memory[gpu_name]['allocated']
                    final_allocated = after_model_memory[gpu_name]['allocated']
                    memory_used = final_allocated - initial_allocated
                    
                    print(f"\n{gpu_name}:")
                    print(f"  模型占用显存: {format_bytes(memory_used)}")
                    print(f"  当前已分配:   {format_bytes(final_allocated)}")
                    print(f"  当前已保留:   {format_bytes(after_model_memory[gpu_name]['reserved'])}")
        
        # 测试推理
        print("\n" + "=" * 60)
        print("测试推理")
        print("=" * 60)
        
        test_prompt = "def fibonacci(n):"
        print(f"测试提示: {test_prompt}")
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        # 记录推理前的显存
        before_inference_memory = get_gpu_memory_info()
        print_memory_info("推理前", before_inference_memory)
        
        # 执行推理
        print("\n执行推理...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        # 记录推理后的显存
        after_inference_memory = get_gpu_memory_info()
        print_memory_info("推理后", after_inference_memory)
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n生成结果:\n{generated_text}")
        
        # 清理
        del inputs, outputs
        torch.cuda.empty_cache()
        
        # 记录清理后的显存
        after_cleanup_memory = get_gpu_memory_info()
        print_memory_info("清理缓存后", after_cleanup_memory)
        
        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 记录错误时的显存
        error_memory = get_gpu_memory_info()
        print_memory_info("错误时", error_memory)
        
        raise


if __name__ == "__main__":
    try:
        model, tokenizer = test_codellama_34b_4bit()
        
        # 保持模型在内存中，等待用户输入
        print("\n模型已加载，按 Enter 键退出并释放显存...")
        input()
        
        # 清理
        del model, tokenizer
        torch.cuda.empty_cache()
        print("\n已清理模型和显存")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n测试失败: {e}")
        torch.cuda.empty_cache()


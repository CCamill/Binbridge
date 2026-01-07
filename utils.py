"""
Binbridge 工具函数
包含各种辅助功能
"""

import os
import json
import random
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import torch
import numpy as np


# =====================
# 随机种子设置
# =====================

def set_seed(seed: int = 42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[Utils] Random seed set to {seed}")


# =====================
# 日志配置
# =====================

def setup_logging(
    log_dir: str = "./logs",
    log_level: int = logging.INFO,
    log_to_file: bool = True
) -> logging.Logger:
    """配置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建 logger
    logger = logging.getLogger("binbridge")
    logger.setLevel(log_level)
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{timestamp}.log")
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# =====================
# 模型相关工具
# =====================

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0
    }


def get_model_size(model: torch.nn.Module) -> str:
    """获取模型大小 (MB)"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = (param_size + buffer_size) / 1024 / 1024
    return f"{total_size:.2f} MB"


def print_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """打印模型信息"""
    params = count_parameters(model)
    size = get_model_size(model)
    
    print(f"\n{'=' * 50}")
    print(f"{model_name} Information")
    print(f"{'=' * 50}")
    print(f"Total parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,} ({params['trainable_ratio']:.2%})")
    print(f"Frozen parameters:    {params['frozen']:,}")
    print(f"Model size:           {size}")
    print(f"{'=' * 50}\n")


# =====================
# 汇编代码处理工具
# =====================

def normalize_assembly(assembly: str) -> str:
    """
    标准化汇编代码格式
    - 统一空白字符
    - 移除注释
    - 统一大小写
    """
    import re
    
    lines = assembly.strip().split('\n')
    normalized_lines = []
    
    for line in lines:
        # 移除注释
        line = re.sub(r';.*$', '', line)
        line = re.sub(r'#.*$', '', line)
        
        # 统一空白
        line = ' '.join(line.split())
        
        # 跳过空行
        if line.strip():
            normalized_lines.append(line.lower())
    
    return '\n'.join(normalized_lines)


def extract_instructions(assembly: str) -> List[str]:
    """提取汇编指令列表"""
    lines = assembly.strip().split('\n')
    instructions = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('.') and ':' not in line:
            instructions.append(line.split()[0] if line.split() else '')
    
    return [i for i in instructions if i]


def get_instruction_statistics(assembly: str) -> Dict[str, Any]:
    """获取汇编代码统计信息"""
    instructions = extract_instructions(assembly)
    
    # 统计指令频率
    from collections import Counter
    instruction_freq = Counter(instructions)
    
    return {
        "total_instructions": len(instructions),
        "unique_instructions": len(instruction_freq),
        "top_instructions": instruction_freq.most_common(10),
        "has_call": any('call' in i for i in instructions),
        "has_jump": any(i.startswith('j') for i in instructions),
        "has_loop": any(i in ['loop', 'loope', 'loopne'] for i in instructions)
    }


# =====================
# 评估指标
# =====================

def calculate_exact_match(predictions: List[str], targets: List[str]) -> float:
    """计算精确匹配率"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    matches = sum(
        pred.strip().lower() == target.strip().lower()
        for pred, target in zip(predictions, targets)
    )
    
    return matches / len(predictions) if predictions else 0.0


def calculate_token_level_accuracy(
    predictions: List[str],
    targets: List[str],
    tokenizer=None
) -> float:
    """计算 token 级别准确率"""
    total_tokens = 0
    correct_tokens = 0
    
    for pred, target in zip(predictions, targets):
        pred_tokens = pred.lower().replace('_', ' ').split()
        target_tokens = target.lower().replace('_', ' ').split()
        
        for pt, tt in zip(pred_tokens, target_tokens):
            if pt == tt:
                correct_tokens += 1
            total_tokens += 1
        
        # 处理长度不匹配
        total_tokens += abs(len(pred_tokens) - len(target_tokens))
    
    return correct_tokens / total_tokens if total_tokens > 0 else 0.0


def calculate_bleu_score(
    predictions: List[str],
    targets: List[str],
    n: int = 4
) -> float:
    """计算 BLEU 分数 (简化版本)"""
    from collections import Counter
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    total_score = 0.0
    
    for pred, target in zip(predictions, targets):
        pred_tokens = pred.lower().replace('_', ' ').split()
        target_tokens = target.lower().replace('_', ' ').split()
        
        score = 0.0
        for i in range(1, n + 1):
            pred_ngrams = get_ngrams(pred_tokens, i)
            target_ngrams = get_ngrams(target_tokens, i)
            
            overlap = sum((pred_ngrams & target_ngrams).values())
            total = sum(pred_ngrams.values())
            
            if total > 0:
                score += overlap / total
        
        total_score += score / n
    
    return total_score / len(predictions) if predictions else 0.0


class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self):
        self.predictions = []
        self.targets = []
    
    def add(self, prediction: str, target: str):
        """添加一对预测和目标"""
        self.predictions.append(prediction)
        self.targets.append(target)
    
    def add_batch(self, predictions: List[str], targets: List[str]):
        """添加一批预测和目标"""
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def calculate(self) -> Dict[str, float]:
        """计算所有指标"""
        return {
            "exact_match": calculate_exact_match(self.predictions, self.targets),
            "token_accuracy": calculate_token_level_accuracy(self.predictions, self.targets),
            "bleu": calculate_bleu_score(self.predictions, self.targets),
            "num_samples": len(self.predictions)
        }
    
    def reset(self):
        """重置"""
        self.predictions = []
        self.targets = []


# =====================
# 数据处理工具
# =====================

def split_dataset(
    data: List[Any],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[List, List, List]:
    """划分数据集"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    if shuffle:
        random.seed(seed)
        data = random.sample(data, len(data))
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return data[:train_end], data[train_end:val_end], data[val_end:]


def save_json(data: Any, path: str, indent: int = 2):
    """保存 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =====================
# GPU 工具
# =====================

def get_gpu_memory_info() -> Dict[str, str]:
    """获取 GPU 内存信息"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        
        info[f"GPU {i} ({props.name})"] = {
            "total": f"{total:.2f} GB",
            "allocated": f"{allocated:.2f} GB",
            "cached": f"{cached:.2f} GB",
            "free": f"{total - allocated:.2f} GB"
        }
    
    return info


def clear_gpu_cache():
    """清理 GPU 缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[Utils] GPU cache cleared")


# =====================
# 进度跟踪
# =====================

class ProgressTracker:
    """训练进度跟踪器"""
    
    def __init__(self, total_steps: int, log_interval: int = 100):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
        self.metrics_history = []
    
    def step(self, metrics: Dict[str, float]):
        """记录一步"""
        self.current_step += 1
        self.metrics_history.append({
            "step": self.current_step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        })
        
        if self.current_step % self.log_interval == 0:
            progress = self.current_step / self.total_steps * 100
            print(f"Progress: {progress:.1f}% ({self.current_step}/{self.total_steps})")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.metrics_history:
            return {}
        
        # 计算平均指标
        keys = [k for k in self.metrics_history[0].keys() if k not in ['step', 'timestamp']]
        averages = {}
        
        for key in keys:
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                averages[f"avg_{key}"] = sum(values) / len(values)
        
        return {
            "total_steps": self.current_step,
            "metrics": averages,
            "history": self.metrics_history
        }
    
    def save(self, path: str):
        """保存进度"""
        save_json(self.get_summary(), path)


if __name__ == "__main__":
    # 测试工具函数
    print("Testing utility functions...")
    
    # 测试随机种子
    set_seed(42)
    
    # 测试汇编处理
    sample_asm = """
    push rbp        ; save base pointer
    mov rbp, rsp
    sub rsp, 0x20
    call some_func
    leave
    ret
    """
    
    normalized = normalize_assembly(sample_asm)
    print(f"\nNormalized assembly:\n{normalized}")
    
    stats = get_instruction_statistics(sample_asm)
    print(f"\nInstruction statistics: {stats}")
    
    # 测试评估指标
    predictions = ["calculate_sum", "init_array", "process_data"]
    targets = ["calculate_sum", "initialize_array", "process_data"]
    
    em = calculate_exact_match(predictions, targets)
    ta = calculate_token_level_accuracy(predictions, targets)
    
    print(f"\nExact match: {em:.2%}")
    print(f"Token accuracy: {ta:.2%}")

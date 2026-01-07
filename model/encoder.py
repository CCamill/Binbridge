"""
汇编代码编码器模块
实现具有重叠策略的分层滑动窗口算法
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from transformers import AutoModel, AutoTokenizer, AutoConfig


class SlidingWindowEncoder(nn.Module):
    """
    滑动窗口编码器
    
    将长汇编序列切分为多个重叠的子片段，分别编码后拼接形成全局特征矩阵
    """
    
    def __init__(
        self,
        encoder_model_name: str = "bert-base-uncased",
        max_seq_length: int = 512,
        window_size: int = 512,
        stride: int = 256,
        freeze_encoder: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.max_seq_length = max_seq_length
        self.window_size = window_size
        self.stride = stride
        self.overlap_ratio = (window_size - stride) / window_size
        self.device = device
        
        # 加载预训练编码器 (CLAP-ASM 或 BERT 类模型)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # 冻结编码器参数以保留预训练知识
        if freeze_encoder:
            self._freeze_encoder()
        
        # 位置编码补偿层 (用于处理窗口间的位置不连续性)
        self.position_compensation = nn.Linear(self.hidden_size, self.hidden_size)
        
    def _freeze_encoder(self):
        """冻结编码器所有参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"[Encoder] Frozen all {sum(p.numel() for p in self.encoder.parameters())} parameters")
    
    def _compute_num_windows(self, seq_length: int) -> int:
        """
        计算窗口数量
        K = ceil((L - W) / S) + 1
        """
        if seq_length <= self.window_size:
            return 1
        return math.ceil((seq_length - self.window_size) / self.stride) + 1
    
    def _create_windows(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]]]:
        """
        将输入序列切分为重叠窗口
        
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            
        Returns:
            windows_ids: 窗口 token ids 列表
            windows_mask: 窗口 attention mask 列表
            window_positions: 每个窗口在原序列中的位置 (start, end)
        """
        batch_size, seq_length = input_ids.shape
        num_windows = self._compute_num_windows(seq_length)
        
        windows_ids = []
        windows_mask = []
        window_positions = []
        
        for k in range(num_windows):
            start = k * self.stride
            end = min(start + self.window_size, seq_length)
            
            # 提取窗口
            window_ids = input_ids[:, start:end]
            window_mask = attention_mask[:, start:end]
            
            # 如果窗口不足 window_size，进行填充
            if end - start < self.window_size:
                pad_length = self.window_size - (end - start)
                window_ids = torch.nn.functional.pad(
                    window_ids, (0, pad_length), value=self.tokenizer.pad_token_id
                )
                window_mask = torch.nn.functional.pad(
                    window_mask, (0, pad_length), value=0
                )
            
            windows_ids.append(window_ids)
            windows_mask.append(window_mask)
            window_positions.append((start, end))
        
        return windows_ids, windows_mask, window_positions
    
    def _merge_overlapping_features(
        self,
        window_features: List[torch.Tensor],
        window_positions: List[Tuple[int, int]],
        original_length: int
    ) -> torch.Tensor:
        """
        合并重叠窗口的特征，对重叠区域进行平均
        
        Args:
            window_features: 每个窗口的特征 [batch_size, window_size, hidden_size]
            window_positions: 窗口位置列表
            original_length: 原始序列长度
            
        Returns:
            merged_features: [batch_size, original_length, hidden_size]
        """
        batch_size = window_features[0].shape[0]
        hidden_size = window_features[0].shape[-1]
        
        # 初始化累加器和计数器
        feature_sum = torch.zeros(
            batch_size, original_length, hidden_size, 
            device=self.device, dtype=window_features[0].dtype
        )
        count = torch.zeros(
            batch_size, original_length, 1,
            device=self.device, dtype=window_features[0].dtype
        )
        
        for features, (start, end) in zip(window_features, window_positions):
            actual_length = end - start
            feature_sum[:, start:end, :] += features[:, :actual_length, :]
            count[:, start:end, :] += 1
        
        # 避免除零
        count = torch.clamp(count, min=1)
        merged_features = feature_sum / count
        
        return merged_features
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_all_windows: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_length] 汇编指令 token ids
            attention_mask: [batch_size, seq_length]
            return_all_windows: 是否返回所有窗口特征
            
        Returns:
            Dict containing:
                - global_features: [batch_size, seq_length, hidden_size] 全局特征矩阵 H_global
                - pooled_output: [batch_size, hidden_size] 池化输出
                - window_features: (optional) 各窗口特征列表
        """
        batch_size, seq_length = input_ids.shape
        
        # 切分窗口
        windows_ids, windows_mask, window_positions = self._create_windows(
            input_ids, attention_mask
        )
        
        # 对每个窗口进行编码
        window_features = []
        for w_ids, w_mask in zip(windows_ids, windows_mask):
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.encoder(
                    input_ids=w_ids.to(self.device),
                    attention_mask=w_mask.to(self.device)
                )
                # 获取最后一层所有 token 的隐藏状态
                hidden_states = outputs.last_hidden_state
                window_features.append(hidden_states)
        
        # 合并重叠特征，形成全局特征矩阵
        global_features = self._merge_overlapping_features(
            window_features, window_positions, seq_length
        )
        
        # 应用位置补偿
        global_features = self.position_compensation(global_features)
        
        # 计算池化输出 (使用有效位置的平均)
        mask_expanded = attention_mask.unsqueeze(-1).float().to(self.device)
        pooled_output = (global_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        result = {
            "global_features": global_features,  # H_global
            "pooled_output": pooled_output,
            "attention_mask": attention_mask
        }
        
        if return_all_windows:
            result["window_features"] = window_features
            result["window_positions"] = window_positions
        
        return result


class AssemblyTokenizer:
    """
    汇编代码专用分词器
    处理汇编指令的特殊格式
    """
    
    def __init__(
        self,
        base_tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
        self.max_length = max_length
        
        # 添加汇编特殊标记
        special_tokens = {
            "additional_special_tokens": [
                "[INST]",  # 指令开始
                "[/INST]",  # 指令结束
                "[REG]",  # 寄存器
                "[IMM]",  # 立即数
                "[MEM]",  # 内存操作
                "[LABEL]",  # 标签
                "[FUNC]",  # 函数调用
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
    
    def preprocess_assembly(self, assembly_code: str) -> str:
        """
        预处理汇编代码
        - 标准化空白字符
        - 标记特殊元素
        """
        import re
        
        # 标准化空白
        code = re.sub(r'\s+', ' ', assembly_code.strip())
        
        # 标记寄存器 (x86-64)
        registers = r'\b(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r[89]|r1[0-5]|eax|ebx|ecx|edx|esi|edi|ebp|esp)\b'
        code = re.sub(registers, r'[REG]\1', code, flags=re.IGNORECASE)
        
        # 标记十六进制立即数
        code = re.sub(r'\b(0x[0-9a-fA-F]+)\b', r'[IMM]\1', code)
        
        # 标记内存访问
        code = re.sub(r'\[([^\]]+)\]', r'[MEM][\1]', code)
        
        return code
    
    def tokenize(
        self,
        assembly_code: str,
        return_tensors: str = "pt",
        padding: str = "max_length",
        truncation: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        分词汇编代码
        
        注意：不进行截断，保留完整序列供滑动窗口处理
        """
        processed_code = self.preprocess_assembly(assembly_code)
        
        return self.tokenizer(
            processed_code,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length if truncation else None
        )
    
    def batch_tokenize(
        self,
        assembly_codes: List[str],
        return_tensors: str = "pt",
        padding: str = "longest"
    ) -> Dict[str, torch.Tensor]:
        """批量分词"""
        processed_codes = [self.preprocess_assembly(code) for code in assembly_codes]
        
        return self.tokenizer(
            processed_codes,
            return_tensors=return_tensors,
            padding=padding,
            truncation=False
        )


if __name__ == "__main__":
    # 测试编码器
    print("Testing SlidingWindowEncoder...")
    
    # 模拟配置
    encoder = SlidingWindowEncoder(
        encoder_model_name="bert-base-uncased",
        window_size=512,
        stride=256,
        freeze_encoder=True,
        device="cpu"
    )
    
    # 模拟输入 (长序列)
    batch_size = 2
    seq_length = 1500
    
    dummy_input = torch.randint(0, 1000, (batch_size, seq_length))
    dummy_mask = torch.ones(batch_size, seq_length)
    
    # 前向传播
    outputs = encoder(dummy_input, dummy_mask, return_all_windows=True)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Global features shape: {outputs['global_features'].shape}")
    print(f"Number of windows: {len(outputs['window_features'])}")
    print(f"Window positions: {outputs['window_positions']}")

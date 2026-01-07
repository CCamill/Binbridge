"""
Attentive Q-Former 模块
实现注意力查询桥接，将长序列特征压缩为固定数量的语义向量
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
        is_cross_attention: bool = False
    ):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.is_cross_attention = is_cross_attention
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(attention_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """转换维度用于多头注意力计算"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] Query 来源
            encoder_hidden_states: [batch_size, encoder_seq_len, hidden_size] Key/Value 来源 (交叉注意力)
            attention_mask: Query 的 attention mask
            encoder_attention_mask: Encoder 的 attention mask
        """
        # Query 始终来自 hidden_states
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        # 交叉注意力: Key/Value 来自 encoder_hidden_states
        if self.is_cross_attention and encoder_hidden_states is not None:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            # 自注意力: Key/Value 来自 hidden_states
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用 attention mask
        if attention_mask is not None:
            # attention_mask: [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # Softmax 和 Dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 计算上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # 输出投影
        output = self.dense(context_layer)
        
        return output


class QFormerBlock(nn.Module):
    """
    Q-Former Transformer Block
    包含自注意力层和可选的交叉注意力层
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        has_cross_attention: bool = True
    ):
        super().__init__()
        
        self.has_cross_attention = has_cross_attention
        
        # 自注意力层
        self.self_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout_prob,
            is_cross_attention=False
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        
        # 交叉注意力层 (用于从全局特征中提取信息)
        if has_cross_attention:
            self.cross_attention = MultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout_prob,
                is_cross_attention=True
            )
            self.cross_attn_layer_norm = nn.LayerNorm(hidden_size)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(hidden_dropout_prob)
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, num_queries, hidden_size] 查询向量
            encoder_hidden_states: [batch_size, seq_len, hidden_size] 全局特征 H_global
            attention_mask: 查询向量的 mask
            encoder_attention_mask: 全局特征的 mask
        """
        # 1. 自注意力 (查询向量之间的交互)
        self_attn_output = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        hidden_states = self.self_attn_layer_norm(hidden_states + self.dropout(self_attn_output))
        
        # 2. 交叉注意力 (从全局特征中提取信息)
        if self.has_cross_attention and encoder_hidden_states is not None:
            cross_attn_output = self.cross_attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
            hidden_states = self.cross_attn_layer_norm(hidden_states + self.dropout(cross_attn_output))
        
        # 3. FFN
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_layer_norm(hidden_states + ffn_output)
        
        return hidden_states


class QFormer(nn.Module):
    """
    Attentive Q-Former
    
    充当 "信息瓶颈"，将冗长的汇编特征压缩为固定数量的高密度语义向量
    
    Architecture:
        - 可学习查询向量 Q ∈ R^{N × d_q}
        - 多层 Transformer Block (自注意力 + 交叉注意力)
        - 线性投影层 (映射到 LLM 嵌入维度)
    """
    
    def __init__(
        self,
        num_query_tokens: int = 64,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        cross_attention_freq: int = 2,
        encoder_hidden_size: int = 768,
        llm_hidden_size: int = 4096
    ):
        super().__init__()
        
        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size
        
        # 可学习查询嵌入 Q ∈ R^{N × d_q}
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, hidden_size)
        )
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # 输入投影 (如果 encoder 和 Q-Former 维度不同)
        self.encoder_proj = None
        if encoder_hidden_size != hidden_size:
            self.encoder_proj = nn.Linear(encoder_hidden_size, hidden_size)
        
        # Transformer Blocks
        self.layers = nn.ModuleList()
        for layer_idx in range(num_hidden_layers):
            # 每隔 cross_attention_freq 层添加交叉注意力
            has_cross_attention = (layer_idx % cross_attention_freq == 0)
            
            self.layers.append(
                QFormerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    hidden_dropout_prob=hidden_dropout_prob,
                    attention_dropout_prob=attention_dropout_prob,
                    has_cross_attention=has_cross_attention
                )
            )
        
        # 软提示符投影层 (映射到 LLM 嵌入维度)
        self.llm_proj = nn.Linear(hidden_size, llm_hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_before_proj: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            encoder_hidden_states: [batch_size, seq_len, encoder_hidden_size] 
                                   全局特征矩阵 H_global
            encoder_attention_mask: [batch_size, seq_len]
            return_before_proj: 是否返回投影前的特征
            
        Returns:
            Dict containing:
                - soft_prompts: [batch_size, num_queries, llm_hidden_size] 软提示符 P
                - query_output: [batch_size, num_queries, hidden_size] Q-Former 输出 Z
        """
        batch_size = encoder_hidden_states.shape[0]
        
        # 投影 encoder 输出到 Q-Former 维度
        if self.encoder_proj is not None:
            encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
        
        # 扩展查询向量到 batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # 通过所有 Transformer Block
        hidden_states = query_tokens
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
        
        query_output = hidden_states  # Z
        
        # 软提示符投影
        soft_prompts = self.llm_proj(query_output)  # P
        
        result = {
            "soft_prompts": soft_prompts,
            "query_output": query_output
        }
        
        if return_before_proj:
            result["query_output"] = query_output
        
        return result


class AttentionPooling(nn.Module):
    """
    注意力池化层
    用于将可变长度序列池化为固定长度表示
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        """
        batch_size = hidden_states.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        
        # 转换 attention mask 格式
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (1 - attention_mask).bool()
        
        output, _ = self.attention(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        
        return output.squeeze(1)


if __name__ == "__main__":
    # 测试 Q-Former
    print("Testing Q-Former...")
    
    batch_size = 2
    seq_len = 1500
    encoder_hidden_size = 768
    llm_hidden_size = 4096
    num_queries = 64
    
    # 模拟全局特征 H_global
    encoder_hidden_states = torch.randn(batch_size, seq_len, encoder_hidden_size)
    encoder_attention_mask = torch.ones(batch_size, seq_len)
    
    # 创建 Q-Former
    qformer = QFormer(
        num_query_tokens=num_queries,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        cross_attention_freq=2,
        encoder_hidden_size=encoder_hidden_size,
        llm_hidden_size=llm_hidden_size
    )
    
    # 前向传播
    outputs = qformer(
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask
    )
    
    print(f"Input shape: {encoder_hidden_states.shape}")
    print(f"Soft prompts shape: {outputs['soft_prompts'].shape}")
    print(f"Compression ratio: {seq_len} -> {num_queries} ({seq_len/num_queries:.1f}x)")

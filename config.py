"""
Binbridge 配置文件
用于二进制代码函数命名的多模态大语言模型
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class EncoderConfig:
    """汇编编码器配置"""
    # 滑动窗口参数
    max_seq_length: int = 512  # L_max: 预训练模型的最大序列长度
    window_size: int = 512  # W: 窗口大小
    stride: int = 256  # S: 步长
    overlap_ratio: float = 0.5  # γ: 重叠率 (W-S)/W
    
    # 编码器模型
    encoder_model_name: str = "CLAP-ASM"  # 基础编码器
    encoder_hidden_size: int = 768  # 编码器隐藏层维度
    freeze_encoder: bool = True  # 是否冻结编码器参数
    
    # 如果使用 BERT 类模型作为替代
    pretrained_encoder_path: str = "bert-base-uncased"


@dataclass
class QFormerConfig:
    """Q-Former 注意力查询桥接配置"""
    num_query_tokens: int = 64  # N: 可学习查询向量数量
    query_hidden_size: int = 768  # d_q: 查询向量维度
    
    # Transformer Block 配置
    num_hidden_layers: int = 6  # Q-Former 层数
    num_attention_heads: int = 12  # 注意力头数
    intermediate_size: int = 3072  # FFN 中间层维度
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # 交叉注意力配置
    cross_attention_freq: int = 2  # 每隔几层添加交叉注意力


@dataclass
class LLMConfig:
    """大语言模型配置"""
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    llm_hidden_size: int = 3072  # d_llm: LLM 嵌入维度
    
    # QLoRA 配置
    use_qlora: bool = True
    qlora_r: int = 64  # LoRA 秩
    qlora_alpha: int = 16  # LoRA alpha
    qlora_dropout: float = 0.05
    qlora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 量化配置
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"  # NormalFloat 4-bit
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # 混合损失权重
    loss_analysis_weight: float = 0.3  # L_analysis 权重
    loss_name_weight: float = 0.7  # L_name 权重
    
    # 优化器配置
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    
    # 保存与日志
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    bf16: bool = False
    
    # 数据配置
    max_input_length: int = 4096  # 最大输入汇编指令数
    max_output_length: int = 512  # 最大输出长度


@dataclass
class DataConfig:
    """数据配置"""
    train_data_path: str = "./data/train.json"
    eval_data_path: str = "./data/eval.json"
    test_data_path: str = "./data/test.json"
    
    # 思维链数据
    use_cot: bool = True  # 是否使用思维链
    cot_data_path: str = "./data/cot_annotations.json"
    
    # 数据处理
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class InferenceConfig:
    """推理配置"""
    # 生成参数
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1
    
    # 输出过滤
    filter_analysis: bool = True  # 是否过滤 <analysis> 标签
    return_full_output: bool = False  # 是否返回完整输出


@dataclass
class BinbridgeConfig:
    """Binbridge 完整配置"""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    qformer: QFormerConfig = field(default_factory=QFormerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # 项目元信息
    project_name: str = "Binbridge"
    version: str = "1.0.0"
    
    def save(self, path: str):
        """保存配置到文件"""
        import json
        from dataclasses import asdict
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "BinbridgeConfig":
        """从文件加载配置"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            encoder=EncoderConfig(**data.get('encoder', {})),
            qformer=QFormerConfig(**data.get('qformer', {})),
            llm=LLMConfig(**data.get('llm', {})),
            training=TrainingConfig(**data.get('training', {})),
            data=DataConfig(**data.get('data', {})),
            inference=InferenceConfig(**data.get('inference', {}))
        )


def get_default_config() -> BinbridgeConfig:
    """获取默认配置"""
    return BinbridgeConfig()


if __name__ == "__main__":
    # 测试配置
    config = get_default_config()
    print(f"Project: {config.project_name} v{config.version}")
    print(f"Encoder: {config.encoder.encoder_model_name}")
    print(f"Q-Former queries: {config.qformer.num_query_tokens}")
    print(f"LLM: {config.llm.model_name}")

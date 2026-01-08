"""
Binbridge 主模型
整合汇编编码器、Q-Former 和大语言模型的多模态架构

基于论文设计：
1. Sliding Window Encoder: 处理任意长度汇编序列
2. Q-Former: 压缩特征到固定64个查询向量
3. LLM with QLoRA: Llama 3.2-3B-Instruct 生成函数名
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.encoder import SlidingWindowEncoder, AssemblyTokenizer
from model.qformer import QFormer


class Binbridge(nn.Module):
    """
    Binbridge: 二进制代码函数命名的多模态大语言模型
    
    Architecture:
        1. Assembly Encoder: 滑动窗口编码器处理长汇编序列 (frozen)
        2. Q-Former: 注意力查询桥接，压缩到 N=64 个软提示符 (trainable)
        3. LLM: QLoRA 微调 Llama 3.2 生成分析和函数名
        
    Input Format (Llama 3.2):
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_prompt}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Arch: {arch}, Opt: {opt} Assembly Sequence: [Soft Prompts from Q-Former]<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
    Output Format:
        <analysis>推理过程</analysis><n>function_name</n>
    """
    
    def __init__(
        self,
        # Encoder 配置
        encoder_model_name: str = "hustcw/clap-asm",
        encoder_hidden_size: int = 768,
        max_seq_length: int = 2048,
        window_size: int = 2048,
        stride: int = 256,
        freeze_encoder: bool = True,
        # Q-Former 配置
        num_query_tokens: int = 64,
        qformer_hidden_size: int = 768,
        qformer_num_layers: int = 6,
        qformer_num_heads: int = 12,
        qformer_intermediate_size: int = 3072,
        cross_attention_freq: int = 2,
        # LLM 配置
        llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        llm_hidden_size: int = 3072,
        use_qlora: bool = True,
        qlora_r: int = 64,
        qlora_alpha: int = 16,
        qlora_dropout: float = 0.05,
        qlora_target_modules: List[str] = None,
        load_in_4bit: bool = True,
        # 其他
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        self.num_query_tokens = num_query_tokens
        self.llm_hidden_size = llm_hidden_size
        
        # 1. 汇编编码器 (Frozen)
        print("[Binbridge] Loading Assembly Encoder...")
        self.assembly_encoder = SlidingWindowEncoder(
            encoder_model_name=encoder_model_name,
            max_seq_length=max_seq_length,
            window_size=window_size,
            stride=stride,
            freeze_encoder=freeze_encoder,
            device=device
        )
        self.assembly_tokenizer = AssemblyTokenizer(
            base_tokenizer_name=encoder_model_name
        )
        
        # 2. Q-Former (Trainable)
        print("[Binbridge] Initializing Q-Former...")
        self.qformer = QFormer(
            num_query_tokens=num_query_tokens,
            hidden_size=qformer_hidden_size,
            num_hidden_layers=qformer_num_layers,
            num_attention_heads=qformer_num_heads,
            intermediate_size=qformer_intermediate_size,
            cross_attention_freq=cross_attention_freq,
            encoder_hidden_size=encoder_hidden_size,
            llm_hidden_size=llm_hidden_size
        )
        
        # 3. LLM with QLoRA
        print("[Binbridge] Loading LLM...")
        self._load_llm(
            model_name=llm_model_name,
            use_qlora=use_qlora,
            qlora_r=qlora_r,
            qlora_alpha=qlora_alpha,
            qlora_dropout=qlora_dropout,
            qlora_target_modules=qlora_target_modules,
            load_in_4bit=load_in_4bit
        )
        
        print("[Binbridge] Model initialized successfully!")
    
    def _load_llm(
        self,
        model_name: str,
        use_qlora: bool,
        qlora_r: int,
        qlora_alpha: int,
        qlora_dropout: float,
        qlora_target_modules: List[str],
        load_in_4bit: bool
    ):
        """加载并配置 LLM (4-bit NF4 量化 + QLoRA)"""
        
        # 4-bit NF4 量化配置
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # 加载基础模型
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if load_in_4bit else None,
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        
        # 应用 QLoRA (Low-Rank Adapters)
        if use_qlora:
            if load_in_4bit:
                self.llm = prepare_model_for_kbit_training(self.llm)
            
            # 目标模块：所有线性层
            if qlora_target_modules is None:
                qlora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            
            lora_config = LoraConfig(
                r=qlora_r,
                lora_alpha=qlora_alpha,
                lora_dropout=qlora_dropout,
                target_modules=qlora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
    
    def get_input_embeddings(self):
        """获取 LLM 的输入嵌入层"""
        return self.llm.get_input_embeddings()
    
    def _get_system_prompt(self) -> str:
        """
        获取结构化系统提示词 (Llama 3.2 格式)
        
        参考论文 5.2 节系统提示词的结构化工程设计
        """
        return """你是一位精通二进制逆向工程的安全专家。你的任务是根据提供的汇编代码摘要（由Visual Encoder提供）和元信息，推断该函数的最可能的原始名称。

请遵循以下分析步骤（思维链）：
1. 调用约定分析：根据Arch识别参数传递方式（寄存器/栈）。
2. 数据流追踪：识别关键常数、API调用模式及内存访问特征。
3. 功能归纳：忽略编译器引入的调度噪声，提取核心语义。
4. 命名生成：使用snake_case格式，动宾结构。

注意：虽然你需要进行上述分析，但最终输出必须严格遵循以下格式：
<analysis>在此处简要写出你的分析过程</analysis><n>函数名</n>"""
    
    def encode_assembly(
        self,
        assembly_code: Union[str, List[str]],
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        编码汇编代码 -> 软提示符
        
        Pipeline:
        1. Tokenize 汇编代码
        2. 滑动窗口编码 -> H_global
        3. Q-Former 压缩 -> 64个软提示符 P
        
        Args:
            assembly_code: 单个或多个汇编代码字符串
            
        Returns:
            soft_prompts: [batch_size, 64, llm_hidden_size] 软提示符
        """
        # 处理输入
        if isinstance(assembly_code, str):
            assembly_code = [assembly_code]
        
        # 分词
        tokenized = self.assembly_tokenizer.batch_tokenize(assembly_code)
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        # 滑动窗口编码 -> H_global
        encoder_outputs = self.assembly_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        global_features = encoder_outputs["global_features"]
        
        # Q-Former 压缩 -> 软提示符
        qformer_outputs = self.qformer(
            encoder_hidden_states=global_features,
            encoder_attention_mask=attention_mask
        )
        
        if return_dict:
            return {
                "soft_prompts": qformer_outputs["soft_prompts"],
                "query_output": qformer_outputs["query_output"],
                "global_features": global_features
            }
        
        return qformer_outputs["soft_prompts"]
    
    def prepare_inputs_for_generation(
        self,
        soft_prompts: torch.Tensor,
        arch: str = "x86_64",
        opt: str = "O2"
    ) -> Dict[str, torch.Tensor]:
        """
        准备生成输入 (Llama 3.2 格式)
        
        将软提示符与文本嵌入拼接，形成完整的输入序列:
        
        [prefix_embeds] + [soft_prompts] + [suffix_embeds]
        
        其中:
        - prefix: <|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|>
                  <|start_header_id|>user<|end_header_id|>Arch: {arch}, Opt: {opt} Assembly Sequence:
        - soft_prompts: Q-Former 输出的 64 个语义向量
        - suffix: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        注意：soft_prompts 融入 User Content 中，作为 Visual Encoder 的语义表示
        """
        batch_size = soft_prompts.shape[0]
        
        # 系统提示词
        system_prompt = self._get_system_prompt()
        
        # 构建 Llama 3.2 格式的 prefix
        prefix_text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Arch: {arch}, Opt: {opt} Assembly Sequence: "
        )
        
        # 构建 suffix (assistant 开始标记)
        suffix_text = "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Tokenize prefix
        prefix_tokens = self.llm_tokenizer(
            prefix_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
            add_special_tokens=False  # 手动添加了 begin_of_text
        )
        
        # Tokenize suffix
        suffix_tokens = self.llm_tokenizer(
            suffix_text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False
        )
        
        # 移动到设备
        prefix_ids = prefix_tokens["input_ids"].to(self.device)
        suffix_ids = suffix_tokens["input_ids"].to(self.device)
        
        # 获取嵌入
        embed_layer = self.get_input_embeddings()
        prefix_embeds = embed_layer(prefix_ids)
        suffix_embeds = embed_layer(suffix_ids)
        
        # 扩展到 batch size
        if prefix_embeds.shape[0] == 1 and batch_size > 1:
            prefix_embeds = prefix_embeds.expand(batch_size, -1, -1)
            suffix_embeds = suffix_embeds.expand(batch_size, -1, -1)
        
        # 拼接: [prefix] + [soft_prompts] + [suffix]
        inputs_embeds = torch.cat([prefix_embeds, soft_prompts, suffix_embeds], dim=1)
        
        # 创建 attention mask
        total_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(batch_size, total_len, device=self.device)
        
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "prefix_len": prefix_embeds.shape[1],
            "soft_prompt_len": soft_prompts.shape[1],
            "suffix_len": suffix_embeds.shape[1]
        }
    
    def forward(
        self,
        assembly_code: Optional[Union[str, List[str]]] = None,
        soft_prompts: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        arch: str = "x86_64",
        opt: str = "O2",
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        训练模式下，labels 为目标输出序列 (<analysis>...</analysis><n>name</n>)
        """
        batch_size = soft_prompts.shape[0] if soft_prompts is not None else 1
        
        # 如果没有提供软提示符，从汇编代码编码
        if soft_prompts is None and assembly_code is not None:
            encoding_result = self.encode_assembly(assembly_code)
            soft_prompts = encoding_result["soft_prompts"]
        
        if soft_prompts is None:
            raise ValueError("Either assembly_code or soft_prompts must be provided")
        
        batch_size = soft_prompts.shape[0]
        
        # 准备输入格式
        if input_ids is not None:
            # 有额外文本输入 (训练模式)
            text_embeds = self.get_input_embeddings()(input_ids)
            
            # 获取系统提示的嵌入
            system_inputs = self.prepare_inputs_for_generation(soft_prompts, arch, opt)
            
            # 拼接: [system+user prefix] + [soft_prompts] + [user suffix] + [target_text]
            inputs_embeds = torch.cat([
                system_inputs["inputs_embeds"],
                text_embeds
            ], dim=1)
            
            # 更新 attention mask
            prompt_len = system_inputs["inputs_embeds"].shape[1]
            prompt_attention_mask = torch.ones(batch_size, prompt_len, device=self.device)
            if attention_mask is not None:
                attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            else:
                attention_mask = torch.ones(batch_size, inputs_embeds.shape[1], device=self.device)
            
            # 更新 labels (系统提示部分不计算损失)
            if labels is not None:
                prompt_labels = torch.full(
                    (batch_size, prompt_len),
                    -100,
                    device=self.device,
                    dtype=labels.dtype
                )
                labels = torch.cat([prompt_labels, labels], dim=1)
        else:
            # 仅软提示符 (推理模式)
            system_inputs = self.prepare_inputs_for_generation(soft_prompts, arch, opt)
            inputs_embeds = system_inputs["inputs_embeds"]
            attention_mask = system_inputs["attention_mask"]
        
        # LLM 前向传播
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
            "soft_prompts": soft_prompts
        }
    
    @torch.no_grad()
    def generate(
        self,
        assembly_code: Union[str, List[str]],
        arch: str = "x86_64",
        opt: str = "O2",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1,
        filter_analysis: bool = True,
        **kwargs
    ) -> List[str]:
        """
        生成函数名称
        
        Args:
            assembly_code: 汇编代码
            arch: CPU 架构 (x86_64, ARM, etc.)
            opt: 优化级别 (O0, O1, O2, O3)
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling
            top_k: top-k sampling
            do_sample: 是否采样
            num_beams: beam search 数量
            filter_analysis: 是否过滤分析部分，只返回函数名
            
        Returns:
            生成的函数名称列表
        """
        self.eval()
        
        if isinstance(assembly_code, str):
            assembly_code = [assembly_code]
        
        # 编码汇编代码 -> 软提示符
        encoding_result = self.encode_assembly(assembly_code)
        soft_prompts = encoding_result["soft_prompts"]
        
        # 准备输入
        inputs = self.prepare_inputs_for_generation(soft_prompts, arch, opt)
        
        # 生成
        outputs = self.llm.generate(
            inputs_embeds=inputs["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
            **kwargs
        )
        
        # 解码输出
        generated_texts = self.llm_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        
        # 过滤分析部分，只返回函数名
        if filter_analysis:
            generated_texts = [
                self._extract_function_name(text) for text in generated_texts
            ]
        
        return generated_texts
    
    def _extract_function_name(self, text: str) -> str:
        """
        从生成文本中提取函数名
        
        输出格式: <analysis>...</analysis><n>function_name</n>
        
        推理阶段优化 (Output Filtering Strategy):
        模型完整输出 analysis + name，但只返回 <n> 内容给用户
        """
        import re
        
        # 尝试提取 <n> 标签内容
        name_match = re.search(r'<n>(.*?)</n>', text, re.DOTALL)
        if name_match:
            return name_match.group(1).strip()
        
        # 备选：尝试提取 [函数名] 格式
        bracket_match = re.search(r'\[([a-z_][a-z0-9_]*)\]', text, re.IGNORECASE)
        if bracket_match:
            return bracket_match.group(1).strip()
        
        # 如果没有标签，返回最后一行（可能是函数名）
        lines = text.strip().split('\n')
        return lines[-1].strip() if lines else text.strip()
    
    def save_pretrained(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存 Q-Former
        torch.save(
            self.qformer.state_dict(),
            os.path.join(save_path, "qformer.pt")
        )
        
        # 保存 LLM (LoRA 适配器)
        self.llm.save_pretrained(os.path.join(save_path, "llm_adapter"))
        
        # 保存分词器
        self.llm_tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
        
        # 保存配置
        config = {
            "num_query_tokens": self.num_query_tokens,
            "llm_hidden_size": self.llm_hidden_size
        }
        torch.save(config, os.path.join(save_path, "config.pt"))
        
        print(f"[Binbridge] Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs):
        """加载预训练模型"""
        import os
        
        # 加载配置
        config = torch.load(os.path.join(load_path, "config.pt"))
        
        # 创建模型实例
        model = cls(**kwargs)
        
        # 加载 Q-Former
        qformer_state = torch.load(os.path.join(load_path, "qformer.pt"))
        model.qformer.load_state_dict(qformer_state)
        
        # 加载 LoRA 适配器
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(
            model.llm.base_model,
            os.path.join(load_path, "llm_adapter")
        )
        
        print(f"[Binbridge] Model loaded from {load_path}")
        return model


class HybridLoss(nn.Module):
    """
    混合损失函数
    
    根据论文设计 (Image 1):
    L = L_analysis + L_name
    
    原理：
    1. 梯度流向 (Gradient Flow): Loss 基于 <analysis> 和 <n> 同时计算
       - 模型为了更准确预测 <n>，必须先优化 <analysis> 的生成质量
       - 分析过程中的每一个 token 都在修正隐藏层状态
       
    2. 条件概率链: 
       - 无思维链: P(name|asm) - 极其复杂的非线性映射
       - 有思维链: P(name|asm, analysis) - analysis 已提取特征
       - 例如: P(name=md5_init|analysis="contains MD5 constants") ≈ 1
       
    3. Q-Former 协同作用:
       - User Content = [Soft Prompts from Q-Former] + "Analyze this assembly: ..."
       - 思维链将隐式向量特征显式化为自然语言逻辑
    
    可选加权版本：
    L = λ_analysis * L_analysis + λ_name * L_name
    """
    
    def __init__(
        self,
        analysis_weight: float = 1.0,  # 默认相等权重
        name_weight: float = 1.0,
        tokenizer = None,
        use_weighted: bool = False
    ):
        super().__init__()
        
        self.analysis_weight = analysis_weight
        self.name_weight = name_weight
        self.tokenizer = tokenizer
        self.use_weighted = use_weighted
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    
    def _find_tag_positions(
        self,
        input_ids: torch.Tensor,
        analysis_start_ids: List[int],
        analysis_end_ids: List[int],
        name_start_ids: List[int],
        name_end_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        查找 <analysis> 和 <n> 标签的位置
        
        Returns:
            analysis_mask: [batch_size, seq_len] analysis 部分的掩码
            name_mask: [batch_size, seq_len] name 部分的掩码
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        analysis_mask = torch.zeros(batch_size, seq_len, device=device)
        name_mask = torch.zeros(batch_size, seq_len, device=device)
        
        # 简化实现：基于标签 token 查找
        # 实际使用时可以预计算这些位置
        for b in range(batch_size):
            ids = input_ids[b].tolist()
            
            # 查找 <analysis> 区域
            in_analysis = False
            in_name = False
            
            for i, token_id in enumerate(ids):
                # 这里使用简化的标签检测逻辑
                # 实际实现应该使用 tokenizer 编码标签
                if in_analysis:
                    analysis_mask[b, i] = 1.0
                if in_name:
                    name_mask[b, i] = 1.0
        
        return analysis_mask, name_mask
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        analysis_mask: Optional[torch.Tensor] = None,
        name_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算混合损失
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            analysis_mask: [batch_size, seq_len] analysis 部分的掩码
            name_mask: [batch_size, seq_len] name 部分的掩码
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # 展平计算损失
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # 逐 token 损失
        token_losses = self.ce_loss(logits_flat, labels_flat)
        token_losses = token_losses.view(batch_size, seq_len)
        
        # 如果提供了分区掩码，分别计算损失
        if analysis_mask is not None and name_mask is not None and self.use_weighted:
            # Analysis 损失
            analysis_losses = token_losses * analysis_mask.float()
            analysis_loss = analysis_losses.sum() / analysis_mask.sum().clamp(min=1)
            
            # Name 损失
            name_losses = token_losses * name_mask.float()
            name_loss = name_losses.sum() / name_mask.sum().clamp(min=1)
            
            # 混合损失: L = λ_analysis * L_analysis + λ_name * L_name
            total_loss = (
                self.analysis_weight * analysis_loss +
                self.name_weight * name_loss
            )
            
            return {
                "loss": total_loss,
                "analysis_loss": analysis_loss,
                "name_loss": name_loss
            }
        else:
            # 标准损失: L = L_analysis + L_name (隐式相加)
            # 即对整个序列计算交叉熵
            valid_mask = (labels != -100).float()
            total_loss = (token_losses * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            
            return {
                "loss": total_loss,
                "analysis_loss": None,
                "name_loss": None
            }


class LabelMaskGenerator:
    """
    标签掩码生成器
    
    用于生成 analysis 和 name 部分的掩码
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # 预编码标签
        self.analysis_start = tokenizer.encode("<analysis>", add_special_tokens=False)
        self.analysis_end = tokenizer.encode("</analysis>", add_special_tokens=False)
        self.name_start = tokenizer.encode("<n>", add_special_tokens=False)
        self.name_end = tokenizer.encode("</n>", add_special_tokens=False)
    
    def generate_masks(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成 analysis 和 name 部分的掩码
        
        Args:
            input_ids: [batch_size, seq_len]
            
        Returns:
            analysis_mask: [batch_size, seq_len]
            name_mask: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        analysis_mask = torch.zeros(batch_size, seq_len, device=device)
        name_mask = torch.zeros(batch_size, seq_len, device=device)
        
        for b in range(batch_size):
            ids = input_ids[b].tolist()
            
            # 查找 <analysis> 区域
            analysis_start_pos = self._find_subsequence(ids, self.analysis_start)
            analysis_end_pos = self._find_subsequence(ids, self.analysis_end)
            
            if analysis_start_pos >= 0 and analysis_end_pos > analysis_start_pos:
                start = analysis_start_pos + len(self.analysis_start)
                end = analysis_end_pos
                analysis_mask[b, start:end] = 1.0
            
            # 查找 <n> 区域
            name_start_pos = self._find_subsequence(ids, self.name_start)
            name_end_pos = self._find_subsequence(ids, self.name_end)
            
            if name_start_pos >= 0 and name_end_pos > name_start_pos:
                start = name_start_pos + len(self.name_start)
                end = name_end_pos
                name_mask[b, start:end] = 1.0
        
        return analysis_mask, name_mask
    
    def _find_subsequence(self, sequence: List[int], pattern: List[int]) -> int:
        """查找子序列位置"""
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return i
        return -1


if __name__ == "__main__":
    print("Binbridge Model Test")
    print("=" * 50)
    
    print("\nNote: Full model test requires downloading LLM weights.")
    print("Testing individual components instead...\n")
    
    # 测试损失函数
    print("Testing HybridLoss...")
    loss_fn = HybridLoss(analysis_weight=1.0, name_weight=1.0)
    
    batch_size, seq_len, vocab_size = 2, 100, 32000
    dummy_logits = torch.randn(batch_size, seq_len, vocab_size)
    dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss_result = loss_fn(dummy_logits, dummy_labels)
    print(f"Loss: {loss_result['loss'].item():.4f}")
    
    # 测试加权损失
    print("\nTesting weighted HybridLoss...")
    weighted_loss_fn = HybridLoss(
        analysis_weight=0.3, 
        name_weight=0.7,
        use_weighted=True
    )
    
    # 创建模拟掩码
    analysis_mask = torch.zeros(batch_size, seq_len)
    analysis_mask[:, 10:50] = 1.0
    name_mask = torch.zeros(batch_size, seq_len)
    name_mask[:, 60:70] = 1.0
    
    weighted_result = weighted_loss_fn(
        dummy_logits, dummy_labels,
        analysis_mask=analysis_mask,
        name_mask=name_mask
    )
    print(f"Total Loss: {weighted_result['loss'].item():.4f}")
    print(f"Analysis Loss: {weighted_result['analysis_loss'].item():.4f}")
    print(f"Name Loss: {weighted_result['name_loss'].item():.4f}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")

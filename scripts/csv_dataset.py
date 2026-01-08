"""
Binbridge CSV 数据集模块
从CSV文件读取汇编代码和思维链标注数据
"""

import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random
import re


@dataclass
class CSVBinbridgeSample:
    """单个训练样本（CSV格式）"""
    key: str  # 唯一标识
    signature: str  # 函数签名
    function_name: str  # 目标函数名
    src_func: Optional[str] = None  # 源代码
    asm_func: str = ""  # 汇编代码
    opti_level: str = "O2"  # 优化级别
    arch: str = "x86_64"  # CPU 架构
    asm_func_len: int = 0  # 汇编代码长度
    analysis: Optional[str] = None  # 思维链分析 (可选)


class CSVBinbridgeDataset(Dataset):
    """
    Binbridge CSV 训练数据集
    
    从CSV文件读取数据，支持以下字段:
    - key: 唯一标识
    - signature: 函数签名
    - function_name: 目标函数名
    - src_func: 源代码（可选）
    - asm_func: 汇编代码
    - opti_level: 优化级别
    - arch: CPU架构
    - asm_func_len: 汇编代码长度
    """
    
    def __init__(
        self,
        csv_path: str,
        assembly_tokenizer,
        llm_tokenizer,
        max_assembly_length: int = 4096 * 2,
        max_output_length: int = 512,
        use_cot: bool = True,
        cot_data_path: Optional[str] = None,
        augment: bool = False,
        chunk_size: Optional[int] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ):
        """
        初始化数据集
        
        Args:
            csv_path: CSV文件路径
            assembly_tokenizer: 汇编代码分词器
            llm_tokenizer: LLM分词器
            max_assembly_length: 最大汇编代码长度
            use_cot: 是否使用思维链
            cot_data_path: 思维链标注文件路径（JSON格式）
            augment: 是否进行数据增强
            chunk_size: 如果指定，只加载部分数据（用于大数据集）
            start_idx: 起始索引
            end_idx: 结束索引（不包含）
        """
        self.assembly_tokenizer = assembly_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.max_assembly_length = max_assembly_length
        self.use_cot = use_cot
        self.augment = augment
        
        # 加载CSV数据
        print(f"[CSVDataset] Loading CSV from {csv_path}...")
        self.df = self._load_csv(csv_path, chunk_size, start_idx, end_idx)
        
        # 加载思维链标注 (如果有)
        self.cot_annotations = {}
        if use_cot and cot_data_path:
            self.cot_annotations = self._load_cot_data(cot_data_path)
        
        print(f"[CSVDataset] Loaded {len(self.df)} samples")
        if self.cot_annotations:
            print(f"[CSVDataset] Loaded {len(self.cot_annotations)} CoT annotations")
    
    def _load_csv(
        self, 
        csv_path: str, 
        chunk_size: Optional[int] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> pd.DataFrame:
        """加载CSV文件"""
        try:
            # 尝试分块读取（适用于大文件）
            if chunk_size:
                chunks = []
                for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                    chunks.append(chunk)
                    if len(chunks) * chunk_size >= (end_idx or float('inf')):
                        break
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(csv_path)
            
            # 切片数据
            if end_idx:
                df = df.iloc[start_idx:end_idx]
            elif start_idx > 0:
                df = df.iloc[start_idx:]
            
            # 验证必需的列
            required_columns = ['function_name', 'asm_func']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # 填充缺失值
            df['arch'] = df.get('arch', 'x86_64').fillna('x86_64')
            df['opti_level'] = df.get('opti_level', 'O2').fillna('O2')
            df['src_func'] = df.get('src_func', '').fillna('')
            df['asm_func'] = df['asm_func'].fillna('')
            
            # 过滤掉空的汇编代码
            df = df[df['asm_func'].str.strip() != '']
            
            return df.reset_index(drop=True)
        
        except Exception as e:
            raise RuntimeError(f"Error loading CSV file: {e}")
    
    def _load_cot_data(self, cot_path: str) -> Dict[str, str]:
        """加载思维链标注数据"""
        try:
            with open(cot_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 支持两种格式：
            # 1. [{"function_name": "...", "analysis": "..."}, ...]
            # 2. {"function_name": "analysis", ...}
            if isinstance(data, list):
                return {item['function_name']: item['analysis'] for item in data}
            elif isinstance(data, dict):
                return data
            else:
                raise ValueError("CoT data must be a list or dict")
        
        except Exception as e:
            print(f"[CSVDataset] Warning: Failed to load CoT data: {e}")
            return {}
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # 提取数据
        assembly_code = str(row['asm_func'])
        function_name = str(row['function_name'])
        arch = str(row.get('arch', 'x86_64'))
        opt = str(row.get('opti_level', 'O2'))
        src_func = str(row.get('src_func', '')) if 'src_func' in row else None
        
        # 获取思维链分析
        analysis = None
        if self.use_cot:
            # 优先使用样本自带的分析（如果CSV中有analysis列）
            if 'analysis' in row and pd.notna(row['analysis']):
                analysis = str(row['analysis'])
            # 否则从 CoT 标注中查找（使用function_name作为键）
            elif function_name in self.cot_annotations:
                analysis = self.cot_annotations[function_name]
            # 也可以使用key作为键（如果CoT数据使用key）
            elif 'key' in row and str(row['key']) in self.cot_annotations:
                analysis = self.cot_annotations[str(row['key'])]
        
        # 数据增强
        if self.augment:
            assembly_code = self._augment_assembly(assembly_code)
        
        # 构建目标输出
        target_text = self._build_target(analysis, function_name)
        
        # 构建输入提示
        input_prompt = self._build_input_prompt(arch, opt)
        
        return {
            'assembly_code': assembly_code,
            'input_prompt': input_prompt,
            'target_text': target_text,
            'function_name': function_name,
            'analysis': analysis,
            'arch': arch,
            'opt': opt,
            'src_func': src_func,
            'key': str(row.get('key', idx)) if 'key' in row else str(idx)
        }
    
    def _build_target(self, analysis: Optional[str], function_name: str) -> str:
        """
        构建目标输出文本
        
        格式: <analysis>分析内容</analysis><n>函数名</n>
        """
        if analysis:
            return f"<analysis>{analysis}</analysis><n>{function_name}</n>"
        else:
            return f"<n>{function_name}</n>"
    
    def _build_input_prompt(self, arch: str, opt: str) -> str:
        """构建输入提示"""
        return f"Arch: {arch}, Opt: {opt}\nAssembly Sequence: [Input features injected here]"
    
    def _augment_assembly(self, assembly: str) -> str:
        """
        汇编代码数据增强
        - 随机替换寄存器别名
        - 随机移除 nop 指令
        - 随机调整指令顺序 (保持依赖关系)
        """
        lines = assembly.strip().split('\n')
        
        # 移除部分 nop 指令
        if random.random() < 0.3:
            lines = [l for l in lines if 'nop' not in l.lower() or random.random() > 0.5]
        
        return '\n'.join(lines)
    
    def get_sample_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """根据key获取样本"""
        if 'key' not in self.df.columns:
            return None
        
        matches = self.df[self.df['key'] == key]
        if len(matches) == 0:
            return None
        
        idx = matches.index[0]
        return self.__getitem__(idx)


class OllamaCOTGenerator:
    """
    使用本地 Ollama 模型的思维链数据生成器
    
    生成高质量的思维链分析用于训练
    """
    
    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: int = 300
    ):
        """
        初始化Ollama CoT生成器
        
        Args:
            model_name: Ollama模型名称
            base_url: Ollama API基础URL
            timeout: 请求超时时间（秒）
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/generate"
        
        # 测试连接
        self._check_connection()
    
    def _check_connection(self):
        """检查Ollama服务是否可用"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"[OllamaCOTGenerator] Connected to Ollama at {self.base_url}")
            else:
                print(f"[OllamaCOTGenerator] Warning: Ollama returned status {response.status_code}")
        except Exception as e:
            print(f"[OllamaCOTGenerator] Warning: Cannot connect to Ollama: {e}")
            print(f"[OllamaCOTGenerator] Make sure Ollama is running at {self.base_url}")
    
    def generate_cot_prompt(
        self,
        assembly_code: str,
        function_name: str,
        arch: str = "x86_64",
        opt_level: str = "O2",
        prompt_path: str = "tmp/CoT_prompt.md"
    ) -> str:
        """生成用于Ollama模型的提示"""
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
        
        prompt = prompt.format(
            asm_func=assembly_code,
            function_name=function_name,
            arch=arch,
            opt_level=opt_level
        )
        
        return prompt
    
    def generate_analysis(
        self,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        prompt: str = ""
    ) -> Optional[str]:
        """
        调用Ollama模型生成分析
        
        Args:
            temperature: 采样温度
            max_tokens: 最大生成token数
        
        Returns:
            生成的分析文本，失败时返回None
        """
        import requests
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result if result else None
            else:
                print(f"[OllamaCOTGenerator] Error: HTTP {response.status_code}")
                print(f"[OllamaCOTGenerator] Response: {response.text}")
                return None
        
        except requests.exceptions.Timeout:
            print(f"[OllamaCOTGenerator] Timeout after {self.timeout}s")
            return None
        except Exception as e:
            print(f"[OllamaCOTGenerator] Error: {e}")
            return None
    
    def batch_generate_from_csv(
        self,
        csv_path: str,
        output_path: str,
        max_samples: Optional[int] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        skip_existing: bool = True,
        batch_size: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        prompt_path: str = "tmp/CoT_prompt.md"
    ):
        """
        从CSV文件批量生成思维链标注
        
        Args:
            csv_path: 输入CSV文件路径
            output_path: 输出JSON文件路径
            max_samples: 最大处理样本数
            start_idx: 起始索引
            end_idx: 结束索引
            skip_existing: 是否跳过已有标注的样本
            batch_size: 批处理大小（目前Ollama API不支持真正的批处理，这里用于进度显示）
            temperature: 采样温度
            max_tokens: 最大生成token数
            prompt_path: 提示词文件路径
        """
        import requests
        from tqdm import tqdm
        
        # 加载CSV
        print(f"[OllamaCOTGenerator] Loading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # 切片
        if end_idx:
            df = df.iloc[start_idx:end_idx]
        elif start_idx > 0:
            df = df.iloc[start_idx:]
        
        if max_samples:
            df = df.head(max_samples)
        
        # 加载已有标注（如果存在）
        existing_annotations = {}
        if skip_existing and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        existing_annotations = {item['key']: item for item in existing_data}
                    elif isinstance(existing_data, dict):
                        existing_annotations = existing_data
            except:
                pass
        
        results = []
        failed_count = 0
        
        # 批量生成
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating CoT"):
            key = str(row.get('key', idx))
            
            # 跳过已有标注
            if skip_existing and key in existing_annotations:
                results.append(existing_annotations[key])
                continue
            
            # 提取数据
            assembly_code = str(row['asm_func'])
            function_name = str(row['function_name'])
            arch = str(row.get('arch', 'x86_64'))
            opt_level = str(row.get('opti_level', 'O2'))

            prompt = self.generate_cot_prompt(
                assembly_code=assembly_code,
                function_name=function_name,
                arch=arch,
                opt_level=opt_level,
                prompt_path=prompt_path
            )
            
            # 生成分析
            result = self.generate_analysis(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            analysis = result.get('cot_label', '').strip()
            function_name = result.get('refined_name', '').strip()
            if analysis:
                results.append({
                    'key': key,
                    'function_name': function_name,
                    'analysis': analysis,
                    'arch': arch,
                    'opt_level': opt_level
                })
            else:
                failed_count += 1
                print(f"[OllamaCOTGenerator] Failed to generate analysis for key={key}")
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"[OllamaCOTGenerator] Generated {len(results)} annotations")
        print(f"[OllamaCOTGenerator] Failed: {failed_count}")
        print(f"[OllamaCOTGenerator] Saved to {output_path}")
        
        return results


def create_csv_dataloader(
    dataset: CSVBinbridgeDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """创建CSV数据集的DataLoader"""
    from data.dataset import collate_fn
    
    def custom_collate(batch):
        return collate_fn(
            batch,
            assembly_tokenizer=dataset.assembly_tokenizer,
            llm_tokenizer=dataset.llm_tokenizer,
            max_assembly_length=dataset.max_assembly_length,
            max_output_length=dataset.max_output_length
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate
    )


if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="CSV Dataset and Ollama CoT Generator")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # CoT生成命令
    cot_parser = subparsers.add_parser("generate_cot", help="Generate CoT annotations using Ollama")
    cot_parser.add_argument("--csv_path", type=str, required=True, help="Input CSV file path")
    cot_parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    cot_parser.add_argument("--model", type=str, default="llama3", help="Ollama model name")
    cot_parser.add_argument("--base_url", type=str, default="http://localhost:11434", help="Ollama API base URL")
    cot_parser.add_argument("--max_samples", type=int, help="Maximum samples to process")
    cot_parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    cot_parser.add_argument("--end_idx", type=int, help="End index")
    cot_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    cot_parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    if args.command == "generate_cot":
        generator = OllamaCOTGenerator(
            model_name=args.model,
            base_url=args.base_url
        )
        generator.batch_generate_from_csv(
            csv_path=args.csv_path,
            output_path=args.output,
            max_samples=args.max_samples,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    else:
        parser.print_help()


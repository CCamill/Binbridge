"""
Binbridge 数据准备脚本
用于从二进制文件提取汇编代码并生成训练数据
"""

import os
import json
import argparse
import subprocess
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class FunctionInfo:
    """函数信息"""
    name: str
    address: str
    assembly: str
    size: int
    architecture: str
    binary_path: str


class BinaryDisassembler:
    """
    二进制反汇编器
    
    支持多种反汇编工具:
    - objdump (GNU Binutils)
    - radare2
    - Ghidra (需要 analyzeHeadless)
    """
    
    def __init__(self, tool: str = "objdump"):
        self.tool = tool
        self._check_tool_available()
    
    def _check_tool_available(self):
        """检查工具是否可用"""
        try:
            if self.tool == "objdump":
                subprocess.run(["objdump", "--version"], capture_output=True, check=True)
            elif self.tool == "radare2":
                subprocess.run(["r2", "-v"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(f"Tool {self.tool} not found. Please install it first.")
    
    def disassemble(self, binary_path: str) -> List[FunctionInfo]:
        """反汇编二进制文件"""
        if self.tool == "objdump":
            return self._disassemble_objdump(binary_path)
        elif self.tool == "radare2":
            return self._disassemble_radare2(binary_path)
        else:
            raise ValueError(f"Unsupported tool: {self.tool}")
    
    def _disassemble_objdump(self, binary_path: str) -> List[FunctionInfo]:
        """使用 objdump 反汇编"""
        # 获取架构信息
        arch_result = subprocess.run(
            ["objdump", "-f", binary_path],
            capture_output=True, text=True
        )
        
        architecture = "unknown"
        if "x86-64" in arch_result.stdout or "x86_64" in arch_result.stdout:
            architecture = "x86_64"
        elif "i386" in arch_result.stdout:
            architecture = "x86"
        elif "arm" in arch_result.stdout.lower():
            architecture = "arm"
        
        # 反汇编
        result = subprocess.run(
            ["objdump", "-d", "-M", "intel", binary_path],
            capture_output=True, text=True
        )
        
        return self._parse_objdump_output(
            result.stdout, 
            architecture, 
            binary_path
        )
    
    def _parse_objdump_output(
        self, 
        output: str, 
        architecture: str,
        binary_path: str
    ) -> List[FunctionInfo]:
        """解析 objdump 输出"""
        functions = []
        current_function = None
        current_assembly = []
        
        # 匹配函数头: 0000000000001149 <main>:
        func_pattern = re.compile(r'^([0-9a-fA-F]+)\s+<([^>]+)>:')
        # 匹配指令行
        inst_pattern = re.compile(r'^\s*([0-9a-fA-F]+):\s+(.+)$')
        
        for line in output.split('\n'):
            func_match = func_pattern.match(line)
            
            if func_match:
                # 保存上一个函数
                if current_function and current_assembly:
                    functions.append(FunctionInfo(
                        name=current_function[1],
                        address=current_function[0],
                        assembly='\n'.join(current_assembly),
                        size=len(current_assembly),
                        architecture=architecture,
                        binary_path=binary_path
                    ))
                
                current_function = (func_match.group(1), func_match.group(2))
                current_assembly = []
            
            elif current_function:
                inst_match = inst_pattern.match(line)
                if inst_match:
                    # 提取指令部分 (去除地址和字节码)
                    inst_line = inst_match.group(2)
                    # 移除字节码，只保留助记符
                    parts = inst_line.split('\t')
                    if len(parts) >= 2:
                        instruction = '\t'.join(parts[1:]).strip()
                        current_assembly.append(instruction)
        
        # 保存最后一个函数
        if current_function and current_assembly:
            functions.append(FunctionInfo(
                name=current_function[1],
                address=current_function[0],
                assembly='\n'.join(current_assembly),
                size=len(current_assembly),
                architecture=architecture,
                binary_path=binary_path
            ))
        
        return functions
    
    def _disassemble_radare2(self, binary_path: str) -> List[FunctionInfo]:
        """使用 radare2 反汇编"""
        # 基本命令序列
        commands = [
            "aaa",  # 分析
            "afl",  # 列出函数
        ]
        
        # 获取函数列表
        result = subprocess.run(
            ["r2", "-q", "-c", ";".join(commands), binary_path],
            capture_output=True, text=True
        )
        
        functions = []
        # 解析函数列表并获取每个函数的反汇编
        for line in result.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 4:
                addr = parts[0]
                name = parts[-1]
                
                # 获取函数反汇编
                asm_result = subprocess.run(
                    ["r2", "-q", "-c", f"s {addr}; pdf", binary_path],
                    capture_output=True, text=True
                )
                
                functions.append(FunctionInfo(
                    name=name,
                    address=addr,
                    assembly=asm_result.stdout,
                    size=len(asm_result.stdout.split('\n')),
                    architecture="unknown",
                    binary_path=binary_path
                ))
        
        return functions


class DatasetBuilder:
    """
    数据集构建器
    
    从二进制文件集合构建训练数据集
    """
    
    def __init__(
        self,
        disassembler: BinaryDisassembler,
        min_function_size: int = 5,
        max_function_size: int = 5000,
        exclude_patterns: List[str] = None
    ):
        self.disassembler = disassembler
        self.min_function_size = min_function_size
        self.max_function_size = max_function_size
        self.exclude_patterns = exclude_patterns or [
            r'^_.*',  # 内部函数
            r'^__.*',  # 编译器生成函数
            r'.*@plt$',  # PLT 条目
            r'^\..*',  # 节
        ]
    
    def _should_include(self, func: FunctionInfo) -> bool:
        """检查是否应该包含该函数"""
        # 检查大小
        if func.size < self.min_function_size:
            return False
        if func.size > self.max_function_size:
            return False
        
        # 检查排除模式
        for pattern in self.exclude_patterns:
            if re.match(pattern, func.name):
                return False
        
        return True
    
    def _clean_function_name(self, name: str) -> str:
        """清理函数名"""
        # 移除 C++ 修饰
        # 这是简化版本，实际可能需要 c++filt
        clean_name = name
        
        # 移除前缀
        prefixes = ['_Z', '__']
        for prefix in prefixes:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        
        return clean_name
    
    def process_binary(self, binary_path: str) -> List[Dict]:
        """处理单个二进制文件"""
        try:
            functions = self.disassembler.disassemble(binary_path)
        except Exception as e:
            print(f"Error processing {binary_path}: {e}")
            return []
        
        samples = []
        for func in functions:
            if not self._should_include(func):
                continue
            
            samples.append({
                "assembly": func.assembly,
                "function_name": self._clean_function_name(func.name),
                "arch": func.architecture,
                "opt": "unknown",  # 优化级别需要额外信息
                "source_binary": os.path.basename(binary_path),
                "address": func.address
            })
        
        return samples
    
    def build_dataset(
        self,
        binary_dir: str,
        output_path: str,
        num_workers: int = 4,
        extensions: List[str] = None
    ):
        """
        从目录构建数据集
        
        Args:
            binary_dir: 二进制文件目录
            output_path: 输出 JSON 路径
            num_workers: 并行工作进程数
            extensions: 要处理的文件扩展名
        """
        if extensions is None:
            extensions = ['', '.so', '.dll', '.exe', '.o', '.a']
        
        # 收集二进制文件
        binary_files = []
        for root, dirs, files in os.walk(binary_dir):
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in extensions or not ext:
                    binary_files.append(os.path.join(root, file))
        
        print(f"[DatasetBuilder] Found {len(binary_files)} binary files")
        
        # 并行处理
        all_samples = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.process_binary, bf): bf 
                for bf in binary_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                samples = future.result()
                all_samples.extend(samples)
        
        print(f"[DatasetBuilder] Extracted {len(all_samples)} function samples")
        
        # 保存数据集
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        print(f"[DatasetBuilder] Dataset saved to {output_path}")
        
        return all_samples


class COTAnnotator:
    """
    思维链标注器
    
    使用 GPT-4 为训练数据添加思维链分析
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        
        if self.api_key:
            import openai
            openai.api_key = self.api_key
    
    def _generate_prompt(self, sample: Dict) -> str:
        """生成标注提示"""
        return f"""作为二进制逆向工程专家，请分析以下汇编代码并生成结构化的思维链分析。

架构: {sample.get('arch', 'x86_64')}
优化级别: {sample.get('opt', 'O2')}
目标函数名: {sample['function_name']}

汇编代码:
```
{sample['assembly'][:2000]}  # 截断以适应上下文
```

请按照以下三个阶段进行分析:

1. **架构与约定分析**: 识别调用约定和参数传递方式
2. **数据流与特征提取**: 定位关键常量、API 调用、内存操作模式
3. **功能归纳**: 综合上述信息，用一句话描述函数功能

请直接输出分析文本，不要使用 markdown 格式。"""
    
    def annotate_sample(self, sample: Dict) -> Optional[str]:
        """为单个样本生成标注"""
        if not self.api_key:
            # 无 API 时返回模板
            return self._generate_template_analysis(sample)
        
        try:
            import openai
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": self._generate_prompt(sample)}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"API error: {e}")
            return None
    
    def _generate_template_analysis(self, sample: Dict) -> str:
        """生成模板分析 (无 API 时使用)"""
        arch = sample.get('arch', 'x86_64')
        name = sample['function_name']
        
        # 简单的启发式分析
        analysis_parts = []
        
        # 架构分析
        if arch == "x86_64":
            analysis_parts.append(
                "This is x86_64 code using the System V AMD64 ABI. "
                "Parameters are passed in rdi, rsi, rdx, rcx, r8, r9."
            )
        elif arch == "arm":
            analysis_parts.append(
                "This is ARM code. Parameters are passed in r0-r3."
            )
        
        # 基于函数名的简单分析
        if "init" in name.lower():
            analysis_parts.append(
                "The function name suggests initialization logic."
            )
        elif "calc" in name.lower() or "compute" in name.lower():
            analysis_parts.append(
                "The function name suggests computation or calculation."
            )
        elif "get" in name.lower() or "set" in name.lower():
            analysis_parts.append(
                "The function name suggests getter/setter pattern."
            )
        
        analysis_parts.append(
            f"Based on the assembly patterns, this function likely implements {name.replace('_', ' ')} functionality."
        )
        
        return " ".join(analysis_parts)
    
    def batch_annotate(
        self,
        input_path: str,
        output_path: str,
        max_samples: int = None,
        skip_existing: bool = True
    ):
        """批量标注数据集"""
        # 加载数据
        with open(input_path, 'r') as f:
            samples = json.load(f)
        
        if max_samples:
            samples = samples[:max_samples]
        
        # 标注
        annotated_samples = []
        for sample in tqdm(samples, desc="Annotating"):
            if skip_existing and sample.get('analysis'):
                annotated_samples.append(sample)
                continue
            
            analysis = self.annotate_sample(sample)
            if analysis:
                sample['analysis'] = analysis
            
            annotated_samples.append(sample)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotated_samples, f, indent=2, ensure_ascii=False)
        
        print(f"[COTAnnotator] Annotated {len(annotated_samples)} samples")


def main():
    parser = argparse.ArgumentParser(description="Prepare Binbridge training data")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # 反汇编命令
    disasm_parser = subparsers.add_parser("disassemble", help="Disassemble binaries")
    disasm_parser.add_argument("--input_dir", type=str, required=True, help="Binary directory")
    disasm_parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    disasm_parser.add_argument("--tool", type=str, default="objdump", help="Disassembly tool")
    disasm_parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    
    # 标注命令
    annotate_parser = subparsers.add_parser("annotate", help="Add CoT annotations")
    annotate_parser.add_argument("--input", type=str, required=True, help="Input JSON path")
    annotate_parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    annotate_parser.add_argument("--max_samples", type=int, help="Max samples to annotate")
    annotate_parser.add_argument("--api_key", type=str, help="OpenAI API key")
    
    # 拆分命令
    split_parser = subparsers.add_parser("split", help="Split dataset")
    split_parser.add_argument("--input", type=str, required=True, help="Input JSON path")
    split_parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    split_parser.add_argument("--train_ratio", type=float, default=0.8)
    split_parser.add_argument("--val_ratio", type=float, default=0.1)
    split_parser.add_argument("--test_ratio", type=float, default=0.1)
    
    args = parser.parse_args()
    
    if args.command == "disassemble":
        disassembler = BinaryDisassembler(tool=args.tool)
        builder = DatasetBuilder(disassembler)
        builder.build_dataset(
            binary_dir=args.input_dir,
            output_path=args.output,
            num_workers=args.workers
        )
    
    elif args.command == "annotate":
        annotator = COTAnnotator(api_key=args.api_key)
        annotator.batch_annotate(
            input_path=args.input,
            output_path=args.output,
            max_samples=args.max_samples
        )
    
    elif args.command == "split":
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import split_dataset, save_json, load_json
        
        data = load_json(args.input)
        train, val, test = split_dataset(
            data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        os.makedirs(args.output_dir, exist_ok=True)
        save_json(train, os.path.join(args.output_dir, "train.json"))
        save_json(val, os.path.join(args.output_dir, "val.json"))
        save_json(test, os.path.join(args.output_dir, "test.json"))
        
        print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

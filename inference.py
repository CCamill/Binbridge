"""
Binbridge 推理脚本
实现输出过滤策略的端到端推理
"""

import os
import sys
import json
import argparse
import re
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BinbridgeConfig, get_default_config
from model.binbridge import Binbridge


@dataclass
class InferenceResult:
    """推理结果"""
    function_name: str  # 预测的函数名
    analysis: Optional[str]  # 分析过程 (可选)
    confidence: float  # 置信度
    raw_output: str  # 原始输出


class BinbridgeInference:
    """
    Binbridge 推理引擎
    
    特性:
    - 批量推理
    - 输出过滤策略
    - 多种输出格式
    - 置信度估计
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[BinbridgeConfig] = None,
        device: str = "cuda"
    ):
        self.config = config or get_default_config()
        self.device = device
        
        # 加载模型
        print(f"[Inference] Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print("[Inference] Model loaded successfully!")
    
    def _load_model(self, model_path: str) -> Binbridge:
        """加载预训练模型"""
        # 这里简化处理，实际需要实现完整的加载逻辑
        model = Binbridge(
            encoder_model_name=self.config.encoder.pretrained_encoder_path,
            llm_model_name=self.config.llm.model_name,
            device=self.device,
            # ... 其他参数
        )
        
        # 加载 Q-Former 权重
        qformer_path = os.path.join(model_path, "qformer.pt")
        if os.path.exists(qformer_path):
            model.qformer.load_state_dict(torch.load(qformer_path))
        
        # 加载 LLM adapter
        adapter_path = os.path.join(model_path, "llm_adapter")
        if os.path.exists(adapter_path):
            from peft import PeftModel
            model.llm = PeftModel.from_pretrained(
                model.llm,
                adapter_path
            )
        
        return model.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        assembly_code: Union[str, List[str]],
        return_analysis: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 256
    ) -> Union[InferenceResult, List[InferenceResult]]:
        """
        预测函数名
        
        Args:
            assembly_code: 汇编代码 (单个或列表)
            return_analysis: 是否返回分析过程
            temperature: 采样温度
            top_p: nucleus sampling 参数
            max_new_tokens: 最大生成 token 数
            
        Returns:
            InferenceResult 或 InferenceResult 列表
        """
        single_input = isinstance(assembly_code, str)
        if single_input:
            assembly_code = [assembly_code]
        
        # 生成
        raw_outputs = self.model.generate(
            assembly_code=assembly_code,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            filter_analysis=False  # 先保留完整输出
        )
        
        # 解析结果
        results = []
        for raw_output in raw_outputs:
            result = self._parse_output(raw_output, return_analysis)
            results.append(result)
        
        if single_input:
            return results[0]
        return results
    
    def _parse_output(
        self,
        raw_output: str,
        return_analysis: bool = False
    ) -> InferenceResult:
        """
        解析模型输出
        
        输出格式: <analysis>...</analysis><n>function_name</n>
        """
        # 提取函数名
        name_match = re.search(r'<n>(.*?)</n>', raw_output, re.DOTALL)
        function_name = name_match.group(1).strip() if name_match else raw_output.strip()
        
        # 提取分析
        analysis = None
        if return_analysis:
            analysis_match = re.search(r'<analysis>(.*?)</analysis>', raw_output, re.DOTALL)
            if analysis_match:
                analysis = analysis_match.group(1).strip()
        
        # 估计置信度 (简化版本)
        confidence = self._estimate_confidence(raw_output, function_name)
        
        return InferenceResult(
            function_name=function_name,
            analysis=analysis,
            confidence=confidence,
            raw_output=raw_output
        )
    
    def _estimate_confidence(self, raw_output: str, function_name: str) -> float:
        """
        估计预测置信度
        
        基于以下启发式规则:
        1. 输出是否包含完整的标签结构
        2. 函数名是否符合命名规范
        3. 分析是否具有合理长度
        """
        confidence = 0.5  # 基础分数
        
        # 检查标签完整性
        if '<n>' in raw_output and '</n>' in raw_output:
            confidence += 0.2
        if '<analysis>' in raw_output and '</analysis>' in raw_output:
            confidence += 0.1
        
        # 检查函数名格式 (snake_case)
        if re.match(r'^[a-z][a-z0-9_]*$', function_name):
            confidence += 0.1
        
        # 检查函数名长度合理性
        if 3 <= len(function_name) <= 50:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def batch_predict(
        self,
        assembly_codes: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
        **kwargs
    ) -> List[InferenceResult]:
        """
        批量预测
        
        Args:
            assembly_codes: 汇编代码列表
            batch_size: 批大小
            show_progress: 是否显示进度条
        """
        results = []
        
        iterator = range(0, len(assembly_codes), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting")
        
        for i in iterator:
            batch = assembly_codes[i:i + batch_size]
            batch_results = self.predict(batch, **kwargs)
            results.extend(batch_results)
        
        return results
    
    def predict_from_file(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 8,
        **kwargs
    ):
        """
        从文件批量预测并保存结果
        
        Args:
            input_path: 输入 JSON 文件路径
            output_path: 输出 JSON 文件路径
            batch_size: 批大小
        """
        # 加载输入数据
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        assembly_codes = [item['assembly'] for item in data]
        
        # 预测
        results = self.batch_predict(
            assembly_codes,
            batch_size=batch_size,
            **kwargs
        )
        
        # 构建输出
        outputs = []
        for item, result in zip(data, results):
            outputs.append({
                "original": item,
                "predicted_name": result.function_name,
                "analysis": result.analysis,
                "confidence": result.confidence,
                "raw_output": result.raw_output
            })
        
        # 保存结果
        with open(output_path, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        
        print(f"[Inference] Results saved to {output_path}")
        
        # 统计信息
        self._print_statistics(results)
    
    def _print_statistics(self, results: List[InferenceResult]):
        """打印统计信息"""
        avg_confidence = sum(r.confidence for r in results) / len(results)
        high_confidence = sum(1 for r in results if r.confidence >= 0.8)
        
        print(f"\n[Statistics]")
        print(f"  Total samples: {len(results)}")
        print(f"  Average confidence: {avg_confidence:.4f}")
        print(f"  High confidence (>=0.8): {high_confidence} ({high_confidence/len(results)*100:.1f}%)")


class InteractivePredictor:
    """
    交互式预测器
    
    用于命令行交互式预测
    """
    
    def __init__(self, inference_engine: BinbridgeInference):
        self.engine = inference_engine
    
    def run(self):
        """运行交互式会话"""
        print("\n" + "=" * 50)
        print("Binbridge Interactive Predictor")
        print("=" * 50)
        print("Enter assembly code (multi-line supported).")
        print("Type 'END' on a new line to submit.")
        print("Type 'quit' to exit.")
        print("=" * 50 + "\n")
        
        while True:
            print("\n[Input Assembly Code]")
            lines = []
            
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                
                if line.strip().lower() == 'quit':
                    print("Goodbye!")
                    return
                
                if line.strip().upper() == 'END':
                    break
                
                lines.append(line)
            
            if not lines:
                continue
            
            assembly_code = '\n'.join(lines)
            
            # 预测
            print("\n[Predicting...]")
            result = self.engine.predict(assembly_code, return_analysis=True)
            
            # 显示结果
            print("\n" + "-" * 40)
            print(f"[Predicted Function Name]")
            print(f"  {result.function_name}")
            print(f"\n[Confidence]")
            print(f"  {result.confidence:.2%}")
            
            if result.analysis:
                print(f"\n[Analysis]")
                print(f"  {result.analysis}")
            
            print("-" * 40)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Binbridge Inference")
    
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--input", type=str, help="Input file path (JSON)")
    parser.add_argument("--output", type=str, help="Output file path (JSON)")
    parser.add_argument("--assembly", type=str, help="Single assembly code for prediction")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for prediction")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--return_analysis", action="store_true", help="Return analysis in output")
    
    args = parser.parse_args()
    
    # 初始化推理引擎
    engine = BinbridgeInference(model_path=args.model_path)
    
    if args.interactive:
        # 交互模式
        predictor = InteractivePredictor(engine)
        predictor.run()
    
    elif args.input and args.output:
        # 批量预测模式
        engine.predict_from_file(
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            temperature=args.temperature,
            return_analysis=args.return_analysis
        )
    
    elif args.assembly:
        # 单个预测模式
        result = engine.predict(
            args.assembly,
            return_analysis=args.return_analysis,
            temperature=args.temperature
        )
        
        print(f"\n[Predicted Function Name]")
        print(f"  {result.function_name}")
        print(f"\n[Confidence]")
        print(f"  {result.confidence:.2%}")
        
        if result.analysis:
            print(f"\n[Analysis]")
            print(f"  {result.analysis}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

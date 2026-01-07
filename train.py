"""
Binbridge 训练脚本
实现蒸馏式思维链训练和 QLoRA 微调
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BinbridgeConfig, get_default_config
from model.binbridge import Binbridge, HybridLoss
from model.encoder import AssemblyTokenizer
from data.dataset import BinbridgeDataset, create_dataloader


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Binbridge 训练器
    
    支持:
    - 混合精度训练
    - 梯度累积
    - 学习率调度
    - 混合损失函数
    - 模型检查点
    """
    
    def __init__(
        self,
        model: Binbridge,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: BinbridgeConfig = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or get_default_config()
        
        # 训练配置
        self.num_epochs = self.config.training.num_epochs
        self.gradient_accumulation_steps = self.config.training.gradient_accumulation_steps
        self.max_grad_norm = self.config.training.max_grad_norm
        self.device = self.config.training.device
        
        # 设置损失函数
        self.loss_fn = HybridLoss(
            analysis_weight=self.config.training.loss_analysis_weight,
            name_weight=self.config.training.loss_name_weight,
            tokenizer=model.llm_tokenizer
        )
        
        # 设置优化器和调度器
        self._setup_optimizer()
        
        # 混合精度
        self.use_amp = self.config.training.fp16 and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # 创建输出目录
        self.output_dir = self.config.training.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 日志
        self.log_file = os.path.join(self.output_dir, "training_log.json")
        self.training_logs = []
    
    def _setup_optimizer(self):
        """配置优化器和学习率调度器"""
        
        # 分离参数组 (不同学习率)
        qformer_params = list(self.model.qformer.parameters())
        llm_params = [p for p in self.model.llm.parameters() if p.requires_grad]
        
        param_groups = [
            {"params": qformer_params, "lr": self.config.training.learning_rate},
            {"params": llm_params, "lr": self.config.training.learning_rate * 0.1}  # LLM 用较小学习率
        ]
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay
        )
        
        # 计算总步数
        num_training_steps = (
            len(self.train_dataloader) * self.num_epochs
            // self.gradient_accumulation_steps
        )
        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_analysis_loss = 0.0
        total_name_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            leave=True
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = self._move_batch_to_device(batch)
            
            # 前向传播 (混合精度)
            if self.use_amp:
                with autocast():
                    outputs = self._forward_step(batch)
                    loss = outputs["loss"] / self.gradient_accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
            else:
                outputs = self._forward_step(batch)
                loss = outputs["loss"] / self.gradient_accumulation_steps
                loss.backward()
            
            # 累积损失
            total_loss += outputs["loss"].item()
            if outputs.get("analysis_loss") is not None:
                total_analysis_loss += outputs["analysis_loss"].item()
            if outputs.get("name_loss") is not None:
                total_name_loss += outputs["name_loss"].item()
            num_batches += 1
            
            # 梯度累积
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{total_loss / num_batches:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # 日志记录
                if self.global_step % self.config.training.logging_steps == 0:
                    self._log_step(epoch, step, total_loss / num_batches)
                
                # 保存检查点
                if self.global_step % self.config.training.save_steps == 0:
                    self._save_checkpoint(epoch, step)
        
        avg_loss = total_loss / num_batches
        avg_analysis_loss = total_analysis_loss / num_batches if total_analysis_loss > 0 else 0
        avg_name_loss = total_name_loss / num_batches if total_name_loss > 0 else 0
        
        return {
            "train_loss": avg_loss,
            "train_analysis_loss": avg_analysis_loss,
            "train_name_loss": avg_name_loss
        }
    
    def _forward_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """单步前向传播"""
        
        # 1. 编码汇编代码
        assembly_input_ids = batch["assembly_input_ids"]
        assembly_attention_mask = batch["assembly_attention_mask"]
        
        # 滑动窗口编码
        encoder_outputs = self.model.assembly_encoder(
            input_ids=assembly_input_ids,
            attention_mask=assembly_attention_mask
        )
        
        # Q-Former 压缩
        qformer_outputs = self.model.qformer(
            encoder_hidden_states=encoder_outputs["global_features"],
            encoder_attention_mask=assembly_attention_mask
        )
        
        soft_prompts = qformer_outputs["soft_prompts"]
        
        # 2. 前向传播 LLM
        model_outputs = self.model(
            soft_prompts=soft_prompts,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        # 3. 计算混合损失
        loss_outputs = self.loss_fn(
            logits=model_outputs["logits"],
            labels=batch["labels"],
            analysis_mask=batch.get("analysis_mask"),
            name_mask=batch.get("name_mask")
        )
        
        return {
            "loss": loss_outputs["loss"],
            "analysis_loss": loss_outputs.get("analysis_loss"),
            "name_loss": loss_outputs.get("name_loss"),
            "logits": model_outputs["logits"]
        }
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """将 batch 数据移动到指定设备"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = self._move_batch_to_device(batch)
            
            outputs = self._forward_step(batch)
            total_loss += outputs["loss"].item()
            
            # 计算准确率 (函数名完全匹配)
            predictions = self._generate_predictions(batch)
            targets = batch["function_names"]
            
            for pred, target in zip(predictions, targets):
                if pred.strip().lower() == target.strip().lower():
                    total_correct += 1
                total_samples += 1
        
        avg_loss = total_loss / len(self.eval_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy
        }
    
    def _generate_predictions(self, batch: Dict) -> list:
        """生成预测结果"""
        # 简化版本，实际需要调用 model.generate
        return ["predicted_func"] * len(batch["function_names"])
    
    def _log_step(self, epoch: int, step: int, loss: float):
        """记录训练日志"""
        log_entry = {
            "epoch": epoch,
            "step": step,
            "global_step": self.global_step,
            "loss": loss,
            "learning_rate": self.scheduler.get_last_lr()[0],
            "timestamp": datetime.now().isoformat()
        }
        self.training_logs.append(log_entry)
        
        # 定期保存日志
        if len(self.training_logs) % 100 == 0:
            with open(self.log_file, 'w') as f:
                json.dump(self.training_logs, f, indent=2)
    
    def _save_checkpoint(self, epoch: int, step: int):
        """保存检查点"""
        checkpoint_dir = os.path.join(
            self.output_dir,
            f"checkpoint-{self.global_step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(checkpoint_dir)
        
        # 保存优化器状态
        torch.save({
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "global_step": self.global_step
        }, os.path.join(checkpoint_dir, "trainer_state.pt"))
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def train(self):
        """完整训练流程"""
        logger.info("=" * 50)
        logger.info("Starting Binbridge Training")
        logger.info("=" * 50)
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Batch size: {self.train_dataloader.batch_size}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info("=" * 50)
        
        for epoch in range(self.num_epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics['train_loss']:.4f}")
            
            # 评估
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                logger.info(f"Epoch {epoch + 1} - Eval Loss: {eval_metrics['eval_loss']:.4f}")
                logger.info(f"Epoch {epoch + 1} - Eval Accuracy: {eval_metrics['eval_accuracy']:.4f}")
                
                # 保存最佳模型
                if eval_metrics['eval_loss'] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics['eval_loss']
                    best_model_dir = os.path.join(self.output_dir, "best_model")
                    self.model.save_pretrained(best_model_dir)
                    logger.info(f"New best model saved to {best_model_dir}")
        
        # 保存最终模型
        final_model_dir = os.path.join(self.output_dir, "final_model")
        self.model.save_pretrained(final_model_dir)
        logger.info(f"Final model saved to {final_model_dir}")
        
        # 保存完整日志
        with open(self.log_file, 'w') as f:
            json.dump(self.training_logs, f, indent=2)
        
        logger.info("Training completed!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train Binbridge Model")
    
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--train_data", type=str, required=True, help="Training data path")
    parser.add_argument("--eval_data", type=str, default=None, help="Evaluation data path")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--use_cot", action="store_true", help="Use chain-of-thought")
    parser.add_argument("--cot_data", type=str, default=None, help="CoT annotation data path")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = BinbridgeConfig.load(args.config)
    else:
        config = get_default_config()
    
    # 更新配置
    config.training.output_dir = args.output_dir
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.data.train_data_path = args.train_data
    config.data.eval_data_path = args.eval_data
    config.data.use_cot = args.use_cot
    config.data.cot_data_path = args.cot_data
    
    # 初始化模型
    logger.info("Initializing model...")
    model = Binbridge(
        encoder_model_name=config.encoder.pretrained_encoder_path,
        encoder_hidden_size=config.encoder.encoder_hidden_size,
        max_seq_length=config.encoder.max_seq_length,
        window_size=config.encoder.window_size,
        stride=config.encoder.stride,
        freeze_encoder=config.encoder.freeze_encoder,
        num_query_tokens=config.qformer.num_query_tokens,
        qformer_hidden_size=config.qformer.query_hidden_size,
        qformer_num_layers=config.qformer.num_hidden_layers,
        qformer_num_heads=config.qformer.num_attention_heads,
        llm_model_name=config.llm.model_name,
        llm_hidden_size=config.llm.llm_hidden_size,
        use_qlora=config.llm.use_qlora,
        qlora_r=config.llm.qlora_r,
        qlora_alpha=config.llm.qlora_alpha,
        load_in_4bit=config.llm.load_in_4bit,
        device=config.training.device
    )
    
    # 创建数据集
    logger.info("Loading datasets...")
    train_dataset = BinbridgeDataset(
        data_path=config.data.train_data_path,
        assembly_tokenizer=model.assembly_tokenizer,
        llm_tokenizer=model.llm_tokenizer,
        use_cot=config.data.use_cot,
        cot_data_path=config.data.cot_data_path
    )
    
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    
    eval_dataloader = None
    if config.data.eval_data_path:
        eval_dataset = BinbridgeDataset(
            data_path=config.data.eval_data_path,
            assembly_tokenizer=model.assembly_tokenizer,
            llm_tokenizer=model.llm_tokenizer,
            use_cot=config.data.use_cot
        )
        eval_dataloader = create_dataloader(
            eval_dataset,
            batch_size=config.training.batch_size,
            shuffle=False
        )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()

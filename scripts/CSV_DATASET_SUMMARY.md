# CSV数据集和Ollama CoT生成器 - 创建总结

## 概述

CSV数据集类和Ollama CoT生成器，支持从CSV文件读取数据，并使用本地Ollama模型生成思维链标注。

## 创建的文件

### 1. `data/csv_dataset.py`
主要模块，包含：
- **CSVBinbridgeDataset**: 从CSV文件读取训练数据的PyTorch Dataset类
- **OllamaCOTGenerator**: 使用本地Ollama模型生成CoT标注的生成器类
- **create_csv_dataloader**: 创建DataLoader的辅助函数

### 2. `scripts/generate_cot_from_csv.py`
命令行脚本，用于批量生成CoT标注：
```bash
python scripts/generate_cot_from_csv.py \
    --csv_path func_pairs_with_strings_train_with_eval.csv \
    --output cot_annotations.json \
    --model llama3 \
    --max_samples 1000
```

### 3. `scripts/test_csv_dataset.py`
测试脚本，用于验证CSV数据集和Ollama连接是否正常工作

### 4. `data/CSV_DATASET_README.md`
详细的使用文档和API参考

## 主要特性

### CSVBinbridgeDataset
- ✅ 支持从CSV文件读取数据（137万条数据）
- ✅ 支持分块加载（处理大数据集）
- ✅ 自动处理缺失值和数据验证
- ✅ 支持思维链标注（从JSON文件加载）
- ✅ 支持数据增强
- ✅ 兼容原有的collate_fn和训练流程

### OllamaCOTGenerator
- ✅ 使用本地Ollama模型（无需API密钥）
- ✅ 支持批量生成CoT标注
- ✅ 支持断点续传（跳过已有标注）
- ✅ 可配置模型、温度、最大tokens等参数
- ✅ 自动处理超时和错误重试

## CSV数据格式要求

您的CSV文件应包含以下列：
- `signature`: 函数签名（可选）
- `function_name`: 目标函数名（**必需**）
- `asm_func`: 汇编代码（**必需**）
- `opti_level`: 优化级别（可选，默认"O2"）
- `arch`: CPU架构（可选，默认"x86_64"）

## 快速开始

### 1. 生成CoT标注

```bash
# 生成前1000个样本的CoT标注
python scripts/generate_cot_from_csv.py \
    --csv_path func_pairs_with_strings_train_with_eval.csv \
    --output cot_annotations.json \
    --model llama3 \
    --max_samples 1000
```

### 2. 使用数据集训练

```python
from data.csv_dataset import CSVBinbridgeDataset, create_csv_dataloader
from transformers import AutoTokenizer

# 初始化分词器
assembly_tokenizer = ...  # 你的汇编代码分词器
llm_tokenizer = AutoTokenizer.from_pretrained("your-llm-model")

# 创建数据集
dataset = CSVBinbridgeDataset(
    csv_path="func_pairs_with_strings_train_with_eval.csv",
    assembly_tokenizer=assembly_tokenizer,
    llm_tokenizer=llm_tokenizer,
    use_cot=True,
    cot_data_path="cot_annotations.json"
)

# 创建DataLoader
dataloader = create_csv_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=True
)

# 训练循环
for batch in dataloader:
    # 使用batch进行训练
    pass
```

## 性能建议

对于137万条数据，建议：

1. **分批生成CoT**: 每批10-50万条，避免单次处理过多数据
2. **使用索引范围**: 使用`--start_idx`和`--end_idx`参数分批处理
3. **并行处理**: 可以启动多个Ollama实例在不同端口，并行生成
4. **使用更快的模型**: 如果对质量要求不是特别高，可以使用较小的模型

## 注意事项

1. **Ollama服务**: 确保Ollama服务正在运行（`ollama serve`）
2. **模型安装**: 确保所需的模型已安装（`ollama pull llama3`）
3. **内存**: 大模型可能需要较多内存，根据实际情况选择模型
4. **超时设置**: 对于长汇编代码，可能需要增加`--timeout`参数

## 下一步

1. 测试CSV数据集加载：`python scripts/test_csv_dataset.py`
2. 生成CoT标注（从小批量开始测试）
3. 使用生成的数据集进行训练

## 问题排查

如果遇到问题，请检查：
- CSV文件路径是否正确
- CSV文件格式是否符合要求
- Ollama服务是否运行
- 模型是否已安装
- 网络连接是否正常（如果使用远程Ollama服务器）

更多详细信息请参考 `data/CSV_DATASET_README.md`。


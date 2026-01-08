# Ollama 网络超时问题解决方案

## 问题描述
```
Error: pull model manifest: Get "https://registry.ollama.ai/v2/library/qwen3-coder/manifests/30b": dial tcp 172.67.182.229:443: i/o timeout
```

## 解决方案

### 方案1: 配置代理（推荐）

如果您的网络需要代理才能访问外网：

```bash
# 设置 HTTP/HTTPS 代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# 或者使用 socks5 代理
export HTTP_PROXY=socks5://your-proxy:port
export HTTPS_PROXY=socks5://your-proxy:port

# 然后重新运行 ollama pull
ollama pull qwen3-coder:30b
```

### 方案2: 使用国内镜像源

配置 Ollama 使用国内镜像：

```bash
# 设置镜像环境变量（如果支持）
export OLLAMA_HOST=https://mirror.example.com  # 替换为实际镜像地址

# 或者修改 Ollama 配置
# 编辑 ~/.ollama/config.json 或相应配置文件
```

### 方案3: 直接使用 Hugging Face 下载模型

由于网络问题，可以直接使用 Hugging Face 下载模型，然后转换为 Ollama 格式：

```bash
# 使用 huggingface-cli 下载
huggingface-cli download Qwen/Qwen2.5-Coder-30B-Instruct --local-dir ./models/qwen3-coder-30b

# 或者使用 Python 脚本下载（见下方）
```

### 方案4: 增加超时时间

```bash
# 设置更长的超时时间
export OLLAMA_TIMEOUT=600  # 10分钟

# 或者使用 curl 测试连接
curl -v --max-time 60 https://registry.ollama.ai
```

### 方案5: 使用本地模型文件

如果您已经有模型文件，可以直接加载：

```bash
# 从本地文件创建模型
ollama create qwen3-coder:30b -f Modelfile

# Modelfile 内容示例：
# FROM /path/to/your/model
```

## 测试网络连接

```bash
# 测试 DNS 解析
nslookup registry.ollama.ai

# 测试 HTTP 连接
curl -I https://registry.ollama.ai

# 测试特定端口
telnet registry.ollama.ai 443
```

## 替代方案：使用 Hugging Face Transformers

如果 Ollama 网络问题无法解决，可以直接使用 Hugging Face 的 transformers 库加载模型（如您创建的测试脚本）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

# 使用 4bit 量化加载
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-30B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
```

这种方式可以：
- 使用 Hugging Face 镜像源（如 hf-mirror.com）
- 支持断点续传
- 更好的错误处理




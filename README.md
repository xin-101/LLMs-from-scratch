# LLMs-from-scratch

从零开始实现大语言模型（LLM），基于 Sebastian Raschka 的 *Build a Large Language Model (From Scratch)* 一书。

## 目录结构

| 章节 | 内容 | 说明 |
|------|------|------|
| ch02 | 数据集准备 | 文本分词（tokenization），构建词汇表，实现 tokenizer |
| ch03 | 编码注意力机制 | 实现自注意力（self-attention）、因果注意力（causal attention）、多头注意力（multi-head attention） |
| ch04 | 从零实现 GPT 模型 | 搭建 GPT-2 架构，包括 Transformer Block、LayerNorm、前馈网络，生成文本 |
| ch05 | 无标签数据上的预训练 | 加载 GPT-2 预训练权重，计算损失（交叉熵/困惑度），实现温度缩放和 top-k 采样 |
| ch06 | 微调分类器 | 在垃圾短信分类任务上微调 GPT-2，实现文本二分类 |
| ch07 | 指令微调 | 基于指令数据微调 GPT-2-medium (355M)，搭建 Chainlit 聊天界面 |

## 环境要求

主要依赖：

- Python 3.10+
- PyTorch >= 2.2.2
- TensorFlow >= 2.18.0（用于加载 GPT-2 预训练权重）
- tiktoken >= 0.5.1
- JupyterLab >= 4.0

安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

每个章节包含一个 Jupyter Notebook（`.ipynb` 文件），在项目根目录下启动 JupyterLab 后即可运行：

```bash
jupyter lab
```

## 注意事项

- 模型权重文件（`.pth`、`.ckpt.*`）因体积过大未被纳入版本控制，请根据各章节代码自行下载
- ch05 和 ch06 使用 GPT-2 124M 模型，ch07 使用 GPT-2 355M 模型，运行前需下载对应的 TensorFlow checkpoint

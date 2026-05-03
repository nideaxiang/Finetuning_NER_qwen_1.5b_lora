# NER_QWEN - 基于 Qwen2 的命名实体识别

基于通义千问 Qwen2-1.5B-Instruct 模型的命名实体识别（NER）项目，使用 LoRA 进行参数高效微调。
源项目：https://github.com/Zeyi-Lin/LLM-Finetune

## 功能特性

- 实体类型：地点、人名、地理实体、组织
- 输出格式：JSON 格式
- 训练方式：LoRA 参数高效微调
- 模型：Qwen2-1.5B-Instruct

## 项目结构

```
NER_QWEN/
├── model.py              # 模型加载
├── data_load.py          # 数据加载与预处理
├── train.py              # 训练脚本
├── predict.py            # 推理脚本
├── metrics.py            # 评估指标
├── ccfbdci.jsonl         # 原始数据集
├── train.jsonl           # 训练集
├── val.jsonl             # 验证集
└── qwen/                 # Qwen2 模型文件
```

## 环境依赖

```bash
pip install torch transformers datasets peft swanlab scikit-learn pandas modelscope
```

## 快速开始

### 1. 训练模型

```bash
python train.py
```

### 2. 推理预测

```bash
python predict.py
```

## 数据格式

原始数据（ccfbdci.jsonl）格式：
```json
{
  "text": "文本内容",
  "entities": [
    {"entity_text": "实体文本", "entity_names": ["实体类型"]}
  ]
}
```

## 输出示例

输入：
```
米洛舍维奇后来在党内的势力可以说是扶摇直上，经历过许多的企业要职，其中还包瓜了出任主要的国营银行总裁。
```

输出：
```json
{"entity_text": "米洛舍维奇", "entity_label": "人名"}
```

## 训练配置

### LoRA 配置
- `task_type`: CAUSAL_LM（因果语言模型）
- `target_modules`: q_proj, k_proj
- `r`: 4（LoRA 秩）
- `lora_alpha`: 16（LoRA 缩放因子）
- `lora_dropout`: 0.1（Dropout 比例）

### 训练参数
- `output_dir`: ./output/Qwen2-NER
- `per_device_train_batch_size`: 1
- `per_device_eval_batch_size`: 1
- `gradient_accumulation_steps`: 16
- `eval_strategy`: steps（每 5 步评估一次）
- `num_train_epochs`: 1
- `save_steps`: 300
- `learning_rate`: 1e-4
- `bf16`: True（混合精度训练）
- `optim`: paged_adamw_32bit（内存优化优化器）
- `gradient_checkpointing`: True（开启梯度检查点）

## 评估函数说明

metrics.py 实现了实体识别的评估逻辑，主要步骤：

1. **解码预测结果**：将 logits 转换为文本
2. **实体提取**：使用正则表达式 `r'entity_text\s*:\s*"(.*?)",\s*entity_type\s*:\s*"(.*?)"'` 从文本中提取实体
3. **指标计算**：
   - 对每类实体（地点、人名、地理实体、组织）分别计算 TP、FP、FN
   - 计算 Precision (P) = TP / (TP + FP)
   - 计算 Recall (R) = TP / (TP + FN)
   - 计算 F1 = 2 × P × R / (P + R)
4. **输出指标**：
   - 各类别的 F1 值：`人名_f1`、`地点_f1`、`地理实体_f1`、`组织_f1`
   - 整体 Micro-F1 值：`f1`、`precision`、`recall`

## 技术栈

- **模型**: Qwen2-1.5B-Instruct
- **微调方法**: LoRA (Low-Rank Adaptation)
- **框架**: HuggingFace Transformers
- **实验管理**: SwanLab

## 许可证

MIT License

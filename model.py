from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import os

model_id = "qwen/Qwen2-1.5B-Instruct"    
model_dir = "./qwen/Qwen2-1___5B-Instruct"

# 在modelscope上下载Qwen模型到本地目录下
if not os.path.exists(model_dir):
    model_dir = snapshot_download(model_id, cache_dir="./", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

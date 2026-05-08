from model import model,tokenizer,model_id,model_dir
from data_load import data_load
from datasets import Dataset
from swanlab.integration.transformers import SwanLabCallback
import swanlab
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
from metrics import compute_metrics

def process_func(example):
    MAX_LEN = 384
    input_ids,attention_mask,labels=[],[],[]
    system_prompt = """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体"."""
    instruction = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids=instruction['input_ids']+response['input_ids']+[tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    #只需要预测 response（即实体的 JSON 内容），而不需要去预测 system_prompt 或 user_input。将这些部分的标签设为 -100
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids)>MAX_LEN:
        input_ids=input_ids[:MAX_LEN]
        attention_mask=attention_mask[:MAX_LEN]
        labels=labels[:MAX_LEN]
    return {'input_ids':input_ids,
        'attention_mask':attention_mask,
        'labels':labels
    }

data_dir="/root/autodl-tmp/NER_QWEN/ccfbdci.jsonl"
process_dir="/root/autodl-tmp/NER_QWEN/ccfbdci_new.jsonl"
train_dir="/root/autodl-tmp/NER_QWEN/train.jsonl"
val_dir="/root/autodl-tmp/NER_QWEN/val.jsonl"
train_df,val_df=data_load(data_dir,process_dir,train_dir,val_dir)



train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
val_dataset = val_ds.map(process_func, remove_columns=val_ds.column_names)
print(len(train_dataset))
print(len(val_dataset))

from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
    ],
    inference_mode=False,#
    r=4,  
    lora_alpha=16,  
    lora_dropout=0.1,  # Dropout 比例
)
model.enable_input_require_grads()
model = get_peft_model(model, config)


#训练
# 修改后的配置建议
args = TrainingArguments(
    output_dir="./output/Qwen2-NER",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,      # 降低评估压力
    gradient_accumulation_steps=16,
    eval_strategy="steps",
    eval_steps=5,  # 临时改为每 5 步评估一次，这样处理 16*5=80 条数据就能看到结果
    logging_steps=1,
    num_train_epochs=1,
    save_steps=300,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    
    # --- 新增的防 OOM 设置 ---
    bf16=True,                         # 混合精度减半显存
    eval_accumulation_steps=1,         # 防止评估时 Logits 堆积
    optim="paged_adamw_32bit",         # 优化器内存优化
    # -----------------------
)


swanlab_callback = SwanLabCallback(
    project="Qwen2-NER-fintune",
    experiment_name="Qwen2-1.5B-Instruct",
    description="使用通义千问Qwen2-1.5B-Instruct模型在NER数据集上微调，实现关键实体识别任务。",
    config={
        "model": model_id,
        "model_dir": model_dir,
        "dataset": "ccfbdci",
        "metrics": compute_metrics,
    },
)

from torch.utils.data import Subset

# 假设你的原始验证集叫 val_dataset
small_eval_dataset = Subset(val_dataset, range(50))

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    compute_metrics=compute_metrics,
    callbacks=[swanlab_callback],
)

train_results = trainer.train()




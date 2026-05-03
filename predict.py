import torch
import swanlab
from model import tokenizer, model

from peft import PeftModel


def predict(messages, model, tokenizer):
    model.eval()
    device = "cuda"
    model.to(device)
    # 1. 使用模板生成文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. 编码并确保所有 tensor 都在正确的设备上
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # 3. 开启无梯度模式
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask, # 建议显式传入
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id, # 确保生成能正常停止
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 4. 只取新生成的部分
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

basemodel=model
lora_adapter_path = "/root/autodl-tmp/output/Qwen2-NER/"

tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, use_fast=False, trust_remote_code=True)
model=PeftModel.from_pretrained(basemodel, model_id=lora_adapter_path)

if __name__ == "__main__":
    # 初始化 SwanLab (如果需要记录)
    swanlab.init(project="Qwen2-NER-Inference", experiment_name="Test-Predict")

    text_data = {
        "instruction": "你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {\"entity_text\": \"南京\", \"entity_label\": \"地理实体\"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出\"没有找到任何实体\". ", 
        "input": "文本:米洛舍维奇后来在党内的势力可以说是扶摇直上，经历过许多的企业要职，其中还包瓜了出任主要的国营银行总裁。"
    }
    
    message = [
        {"role": "system", "content": text_data["instruction"]},
        {"role": "user", "content": text_data["input"]}
    ]

    print(f"正在推理输入: {text_data['input']}")
    
    # 执行推理
    result = predict(message, model, tokenizer)
    
    print(f"推理结果: {result}")
    
    # 记录到 SwanLab
    swanlab.log({
        "Input": swanlab.Text(text_data["input"]), 
        "Prediction": swanlab.Text(result)
    })
    swanlab.finish()

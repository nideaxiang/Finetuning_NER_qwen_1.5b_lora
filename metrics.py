import json
import re
from collections import defaultdict
import numpy as np
import torch
from model import tokenizer


def compute_metrics(eval_preds):
    
    logits, labels = eval_preds
    # 1. 解码
    preds = np.argmax(logits, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 2. 定义每一类的计数器
    # 结构: {"人名": {"tp": 0, "fp": 0, "fn": 0}, "地点": {...}}
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    match_names = ["地点", "人名", "地理实体", "组织"]

    def extract_entities(text):
        entities = set()
        pattern = r'entity_text\s*:\s*"(.*?)",\s*entity_type\s*:\s*"(.*?)"'
        matches = re.findall(pattern, text)
        for name, etype in matches:
            if etype in match_names:
                entities.add((name, etype))
        return entities

    # 3. 核心计算逻辑
    for pred_str, label_str in zip(decoded_preds, decoded_labels):
        pred_entities = extract_entities(pred_str)
        label_entities = extract_entities(label_str)
        
        # 遍历所有可能的类别来计算各类的指标
        for etype in match_names:
            p_sub = {e for e in pred_entities if e[1] == etype}
            l_sub = {e for e in label_entities if e[1] == etype}
            
            class_metrics[etype]["tp"] += len(p_sub & l_sub)
            class_metrics[etype]["fp"] += len(p_sub - l_sub)
            class_metrics[etype]["fn"] += len(l_sub - p_sub)

    # 4. 汇总所有指标
    results = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for etype, counts in class_metrics.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        
        # 计算每一类的 P, R, F1
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        
        # 格式化输出给 SwanLab，例如: eval/人名_f1
        results[f"{etype}_f1"] = f1
        # results[f"{etype}_precision"] = p # 如果不需要太细可以注释掉
        # results[f"{etype}_recall"] = r
        
        # 累加给整体指标（Micro-F1）
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # 5. 计算整体总指标
    final_p = total_tp / (total_tp + total_fp + 1e-10)
    final_r = total_tp / (total_tp + total_fn + 1e-10)
    final_f1 = 2 * final_p * final_r / (final_p + final_r + 1e-10)
    results["f1"] = final_f1
    results["precision"] = final_p
    results["recall"] = final_r
    return results
   

#数据集加载与切分

import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
def data_load(data_dir,process_dir,train_dir,val_dir,split_ratio=0.9):
    messages=[]
    with open(data_dir,'r') as f:
        for line in f:
            data = json.loads(line)
            input =data['text'] #文本
            entity =data['entities'] #实体
            match_names=["地点", "人名", "地理实体", "组织"]

            entity_str=""

            for en in entity:
                en_json=dict(en)
                en_text=en_json['entity_text']
                en_type=en_json['entity_names']
                for name in en_type:
                    if name in match_names:
                        en_label=name
                        break
                entity_str+=f"""{"entity_text"} :"{en_text}",{"entity_type"} :"{en_label}" """
            
            if entity_str=="":
                entity_str="未能识别出目标的实体"
            
            message={
                "instruction":"""你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,
                "input":f"文本:{input}",
                "output":entity_str
            }
            messages.append(message)
    
    with open(process_dir,'w',encoding='utf-8') as f:
        for message in messages:
            f.write(json.dumps(message,ensure_ascii=False)+'\n')

    total_df=pd.read_json(process_dir,lines=True)
    train_df,val_df=train_test_split(total_df,test_size=1-split_ratio,random_state=42)

    if not os.path.exists(train_dir):
        with open(train_dir,'w',encoding='utf-8') as f:
            for message in train_df.to_dict(orient='records'):
                f.write(json.dumps(message,ensure_ascii=False)+'\n')
    if not os.path.exists(val_dir):
        with open(val_dir,'w',encoding='utf-8') as f:
            for message in val_df.to_dict(orient='records'):
                f.write(json.dumps(message,ensure_ascii=False)+'\n')
    
    return train_df,val_df

    

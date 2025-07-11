import json
import random
from collections import defaultdict

def preprocess_content(msg):
    """处理消息内容：非文本转为占位符"""
    if msg["type"] != "文本":
        return f"[{msg['type']}]"
    return str(msg["content"])

def chat2alpaca(conversation):
    """将单个对话转换为Alpaca格式样本"""
    dataset = []
    history = []
    current_instruction = []
    current_output = []
    current_role = None
    
    for msg in conversation["messages"]:
        content = preprocess_content(msg)
        role = msg["role"]
        
        if role == "User":
            if current_role == "AI" and current_output:
                # 保存当前交互对
                instruction_str = "\n".join(current_instruction)
                output_str = "\n".join(current_output)
                
                dataset.append({
                    "instruction": instruction_str,
                    "input":"",
                    "output": output_str,
                    "system": "",
                    "history": [pair.copy() for pair in history]
                })
                
                # 更新历史记录
                history.append([instruction_str, output_str])
                current_instruction = []
                current_output = []
            
            current_instruction.append(content)
            current_role = "User"
        
        elif role == "AI":
            if current_instruction:  # 仅当有用户消息时才处理AI回复
                current_output.append(content)
                current_role = "AI"
    
    # 处理最后一个交互对
    if current_instruction and current_output:
        instruction_str = "\n".join(current_instruction)
        output_str = "\n".join(current_output)
        
        dataset.append({
            "instruction": instruction_str,
            "input": "",
            "output": output_str,
            "system": "",
            "history": [pair.copy() for pair in history],
        })
    
    return dataset

def split_dataset(conversations, ratios=(0.8, 0.1, 0.1), seed=42):
    """
    按比例分割数据集，确保同一对话的样本不跨分割
    返回三个数据集：train, val, test
    """
    assert sum(ratios) == 1.0, "比例总和必须为1"
    assert len(ratios) == 3, "需要三个比例值"
    
    random.seed(seed)
    
    # 只处理有有效样本的对话
    valid_convs = []
    for conv in conversations:
        samples = chat2alpaca(conv)
        if samples:
            valid_convs.append({
                "conversation_id": conv["conversation_id"],
                "samples": samples
            })
    
    # 打乱对话顺序
    random.shuffle(valid_convs)
    
    # 计算分割点
    total_convs = len(valid_convs)
    train_end = int(total_convs * ratios[0])
    val_end = train_end + int(total_convs * ratios[1])
    
    # 分割对话组
    train_convs = valid_convs[:train_end]
    val_convs = valid_convs[train_end:val_end]
    test_convs = valid_convs[val_end:]
    
    # 组装最终数据集（每个分割集为单个列表）
    def collect_samples(conv_list):
        return [s for conv in conv_list for s in conv["samples"]]
    
    return (
        collect_samples(train_convs),
        collect_samples(val_convs),
        collect_samples(test_convs)
    )

def generate_dataset_report(data, name):
    """生成简洁数据集报告"""
    num_samples = len(data)
    
    # 仅保留关键统计信息
    return {
        "name": name,
        "samples": num_samples,
    }

def process_chat_data(input_file, output_prefix, ratios=(0.8, 0.1, 0.1), seed=42):
    """
    主处理函数：加载聊天数据，转换格式，分割并保存为单个文件
    """
    # 加载原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    print(f"加载成功! 共 {len(conversations)} 个对话")
    
    # 转换并分割数据集
    train_data, val_data, test_data = split_dataset(
        conversations, 
        ratios=ratios, 
        seed=seed
    )
    
    # 生成报告
    reports = [
        generate_dataset_report(train_data, "训练集"),
        generate_dataset_report(val_data, "验证集"),
        generate_dataset_report(test_data, "测试集")
    ]
    
    # 保存数据集为单个文件
    with open(f"{output_prefix}_train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(f"{output_prefix}_val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(f"{output_prefix}_test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 保存报告
    with open(f"{output_prefix}_report.json", 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print("\n===== 数据集处理完成 =====")
    print(f"总样本数: {len(train_data) + len(val_data) + len(test_data)}")
    for report in reports:
        print(f"\n{report['name']}报告:")
        print(f"  样本数: {report['samples']}")
    
    print("\n文件已保存:")
    print(f"  训练集: {output_prefix}_train.json")
    print(f"  验证集: {output_prefix}_val.json")
    print(f"  测试集: {output_prefix}_test.json")
    print(f"  统计报告: {output_prefix}_report.json")

if __name__ == "__main__":
    # ===== 配置区域 =====
    INPUT_FILE = "data\cleaned_conversations.json"     # 输入的聊天记录文件
    OUTPUT_PREFIX = "alpaca_dataset"      # 输出文件前缀
    RATIOS = (1.0, 0.0, 0.0)           # 训练/验证/测试比例
    SEED = 42                             # 随机种子
    
    # ===== 执行处理 =====
    process_chat_data(INPUT_FILE, OUTPUT_PREFIX, RATIOS, SEED)
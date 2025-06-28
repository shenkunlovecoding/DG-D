import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 读取Excel文件
df = pd.read_excel('data\mock.xlsx')

# 1. 数据清洗
# --------------------------
# 过滤无效消息类型 (保留文本、图片、引用回复)
valid_types = ['文本', '图片', '引用回复', '动画表情']
df = df[df['type_name'].isin(valid_types)]

# 过滤系统通知和撤回消息
df = df[~df['type_name'].str.contains('系统通知|revokemsg', na=False)]

# 处理媒体文件：图片/表情替换为占位符
media_placeholder = '[图片]'  # 统一的占位符
img_condition = df['type_name'].isin(['图片', '动画表情'])
df.loc[img_condition, 'msg'] = media_placeholder

# 2. 时间处理 - FIXED
# --------------------------
# 检查CreateTime是否是时间戳类型
if pd.api.types.is_datetime64_any_dtype(df['CreateTime']):
    # 如果是，直接使用
    df['timestamp'] = df['CreateTime']
else:
    # 转换微信时间戳(Excel格式)为datetime对象
    # 确保是数字类型
    df['timestamp'] = pd.to_datetime(df['CreateTime'], errors='coerce')
# 3. 对话分割 (按15分钟间隔)
# --------------------------
# 按时间排序
df = df.sort_values(by='timestamp')

# 计算时间差 (转换为分钟)
time_diffs = df['timestamp'].diff().dt.total_seconds() / 60
time_diffs.iloc[0] = 0  # 第一行设为0

# 创建对话ID：当时间差>15分钟时生成新对话ID
conversation_id = (time_diffs > 15).cumsum()
df['conversation_id'] = conversation_id

# 4. 对话有效性过滤
# --------------------------
# 标记发送者身份 (0=用户, 1=AI)
df['role'] = np.where(df['is_sender'] == 1, 'AI', 'User')

# 计算每个对话中的用户/AI消息数量
role_counts = df.groupby('conversation_id')['role'].value_counts().unstack().fillna(0)
valid_convs = role_counts[role_counts['User'] > 0].index  # 确保有用户消息
df = df[df['conversation_id'].isin(valid_convs)]

# 5. 额外过滤：删除只有单方面消息的对话（用户没有回复或AI没有回复）
both_sides = role_counts[(role_counts['AI'] > 0) & (role_counts['User'] > 0)].index
df = df[df['conversation_id'].isin(both_sides)]

# 6. 格式化输出
# --------------------------
final_data = []
for conv_id, group in df.groupby('conversation_id'):
    participants = ['AI', 'User']  # 固定参与者
    messages = []
    
    for _, row in group.iterrows():
        messages.append({
            "time": row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            "role": row['role'],
            "content": row['msg'],
            "type": row['type_name']
        })
    
    final_data.append({
        "conversation_id": int(conv_id),
        "message_count": len(messages),
        "participants": participants,
        "start_time": group['timestamp'].min().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": group['timestamp'].max().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages
    })

# 保存结果到JSON文件
import json
with open('cleaned_conversations.json', 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"处理完成！成功清洗并分割了 {len(final_data)} 个有效对话")
print(f"已保存为 cleaned_conversations.json")
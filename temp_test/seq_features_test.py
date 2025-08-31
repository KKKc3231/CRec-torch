import numpy as np
import torch

# 模拟你的 seq_features 数据（可替换为实际数据）
seq_features = np.array([
    ['6899,6899,6899,6899,6899,6899,474,474,328,328,3162,328,328,328,328,847,328,328,847'],
    ['6899,6899,6899,6899,6899,6899,6899,6899,6899,6899,6899,6899,6899,6899,6899,6899,6899,638,638'],
    ['592,524,144,144,144,144,94,144,52,551,1,758,43,188,18,9,94,407,94']
], dtype=object)

# 1. 拆分字符串并转换为数值列表
def parse_seq(seq_str):
    # 按逗号拆分字符串，转换为 int 或 float（根据需求选，这里用 int 示例）
    return [int(num) for num in seq_str[0].split(',')]

parsed_seqs = [parse_seq(seq) for seq in seq_features]

# 3. 转换为 numpy 数组（这里用 parsed_seqs 直接转，若做了填充用 padded_seqs）
seq_array = np.array(parsed_seqs, dtype=np.int64)  # 按需选 int64/float32 等类型

# 4. 转换为 PyTorch Tensor
seq_tensor = torch.from_numpy(seq_array)

print(seq_tensor)
print(seq_tensor.shape)
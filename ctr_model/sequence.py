import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# dice激活函数
class Dice(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.alpha = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_reshape = x.view(batch_size * seq_len, dim)
        x_norm = self.bn(x_reshape) # 先归一化，再过sigmoid
        x_norm = x_norm.view(batch_size, seq_len, dim)
        x_p = torch.sigmoid(x_norm)
        return self.alpha * (1 - x_p) * x + x_p * x
        
# DIN序列特征提取的attention
class AttentionSequence(nn.Module):
    
    def __init__(self, embedding_dim = 64, att_hidden_units=[80, 40], activation='dice'):
        super().__init__()
        # activation name
        self.activation = activation
        
        # linear && dice 
        self.att_linear_layers = nn.Sequential(
            nn.Linear(embedding_dim * 4, att_hidden_units[0]),
            Dice(att_hidden_units[0]), 
            nn.Linear(att_hidden_units[0], att_hidden_units[-1]),
            Dice(att_hidden_units[-1])
        )
        
        # 用于计算attention score
        self.att_projection = nn.Linear(att_hidden_units[-1], 1)        

    
    def forward(self, query, keys):
        # query: predict item   shape: [batch, 1, embedding_dim]
        # keys: hist item list  shape: [batch, seq_len, embedding_dim]
        item = query
        batch_size, seq_len, embedding_dim = keys.shape
        query = query.expand(-1, seq_len, -1) # [batch, seq_len, embedding_dim]

        # attention input
        att_input = torch.cat([query, keys, query - keys, query * keys], dim=-1)
        
        # attention out && attention weight
        att_out = self.att_linear_layers(att_input)
        score = self.att_projection(att_out).squeeze(dim=-1)
        
        # mask attention
        mask = torch.ne(torch.sum(keys, dim=-1), 0)  # padding_idx的embedding的和为0，通过这个来mask
        att_score = score.masked_fill(~mask, -1e9)
        att_score = att_score - torch.max(att_score, dim=-1, keepdim=True)[0]
        att_weight = F.softmax(att_score, dim=-1).unsqueeze(-1) # 在seq_len维度上softmax

        # print(att_weight)
        user_interest = torch.sum(att_weight * keys, dim=1) 
        
        return user_interest

class DIN(nn.Module):
    def __init__(self, item_counts, max_seq_len=20, embedding_dim=64, hidden_units=[80, 40],
                  activation='dice', dropout=0.2, l2_reg=0.1, padding_idx=None):
        super().__init__()
        # item embedding
        self.item_embedding = nn.Embedding(
            num_embeddings = item_counts + 1, 
            embedding_dim = embedding_dim,
            padding_idx = padding_idx  if padding_idx else None # 不满序列长度的话，padding
        )

        # attention
        self.attention = AttentionSequence(embedding_dim=embedding_dim, att_hidden_units=hidden_units, activation=activation)

    def forward(self, item, seq_list):
        # embedding
        item_embedding = self.item_embedding(item)
        seq_embedding = self.item_embedding(seq_list)

        # seq attention
        user_interest = self.attention(item_embedding, seq_embedding)
        user_interest = torch.nan_to_num(user_interest, nan=0.0) # [padding_idx, padding_idx, ..., padding_idx], 这种是nan，直接全0

        return user_interest

# if __name__ == '__main__':

#     item_counts = 7000
#     padding_id = 6899
#     batch_size = 32
#     seq_max_len = 20
#     embedding_dim = 64

#     # model
#     model = DIN(
#         item_counts = item_counts,
#         max_seq_len = seq_max_len,
#         embedding_dim = 64,
#         hidden_units = [80, 40],
#         activation = 'dice',
#         padding_idx = padding_id
#     )

#     # random item && seq
#     item_ids = torch.randint(0, item_counts, (batch_size, 1))
#     seq_ids = torch.randint(6890, 6900, (batch_size, seq_max_len))  
#     print("i", item_ids.shape)
#     print(seq_ids.shape)

#     user_interest_features = model(item_ids, seq_ids)

#     # seq
#     # print(item_embedding.shape)
#     print(user_interest_features.shape)
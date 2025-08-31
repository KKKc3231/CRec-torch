import torch
import torch.nn as nn
from .sequence import DIN

class ESMMDIN(nn.Module):
    def __init__(self, num_feature_dim, cat_feature_sizes, item_counts, padding_id, item_embedding_dim=64, embedding_dim=6, mlp_dim=32, task_dim=512):
        super().__init__()

        # item_embedding
        item_counts = item_counts # item_counts: 6200多，但padding_idx，数据中说6899，只能扩大embedding表的大小了
        self.din = DIN(item_counts=item_counts, embedding_dim=item_embedding_dim, hidden_units=[80, 40], padding_idx=padding_id)
        # mlp + embedding
        self.num_linear = nn.Linear(num_feature_dim, mlp_dim)
        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(size, embedding_dim) 
            for name, size in cat_feature_sizes.items()
        })
        
        # embedding size
        total_embed_dim = embedding_dim * len(cat_feature_sizes)
        shared_dim = mlp_dim + total_embed_dim + item_embedding_dim # item_embedding_dim: seq_feature_dim

        # CTR Tower
        self.ctr_layer = nn.Sequential(
            nn.Linear(shared_dim, task_dim),
            nn.ReLU(),
            nn.Linear(task_dim, 1),
            nn.Dropout(0.1)
        )
        
        # CVR Tower
        self.cvr_layer = nn.Sequential(
            nn.Linear(shared_dim, task_dim),
            nn.ReLU(),
            nn.Linear(task_dim, 1),
            nn.Dropout(0.1)
        )
    
    def forward(self, num_feats, cat_feats, item_id, seq_id):
        # din features 
        din_out = self.din(item_id, seq_id)
        # mlp
        num_out = self.num_linear(num_feats)
        
        # 处理类别特征
        cat_embeds = []
        for name, embed in self.cat_embeddings.items():
            cat_embeds.append(embed(cat_feats[name]))
        
        # 拼接特征
        cat_out = torch.cat(cat_embeds, dim=1)
        shared_features = torch.cat([num_out, cat_out], dim=1)
        shared_features = torch.cat([shared_features, din_out], dim=1)
        # print("shared_features: ", shared_features.shape)
        
        # 分别计算 CTR 和 CVR
        ctr_logits = self.ctr_layer(shared_features)
        cvr_logits = self.cvr_layer(shared_features)
        
        return ctr_logits, cvr_logits
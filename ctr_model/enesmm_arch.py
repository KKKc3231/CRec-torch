# ESMMMOE
import torch
import torch.nn as nn
import torch.nn.functional as F

class ENESMM(nn.Module):
    def __init__(self, num_feature_dim, cat_feature_sizes, embedding_dim=6, mlp_dim=32, task_dim=512, 
                 num_experts=4, expert_dim=128):
        super().__init__()
        # mlp + embedding
        self.num_linear = nn.Sequential(
            nn.Linear(num_feature_dim, mlp_dim),
            nn.ReLU(),
            nn.LayerNorm(mlp_dim),
            nn.Dropout(0.1)
        )
        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(size, embedding_dim) 
            for name, size in cat_feature_sizes.items()
        })
        
        # embedding size
        total_embed_dim = embedding_dim * len(cat_feature_sizes)
        shared_dim = mlp_dim + total_embed_dim
        
        # MMOE
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_dim, expert_dim),
                nn.LayerNorm(expert_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            for _ in range(num_experts)
        ])
        
        # ctr_gate 
        self.ctr_gate = nn.Sequential(
            nn.Linear(shared_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # cvr_gate
        self.cvr_gate = nn.Sequential(
            nn.Linear(shared_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # CTR Tower
        self.ctr_layer = nn.Sequential(
            nn.Linear(expert_dim, task_dim),
            nn.ReLU(),
            nn.LayerNorm(task_dim),
            nn.Dropout(0.1),
            nn.Linear(task_dim, 1)
        )
        
        # CVR Tower
        self.cvr_layer = nn.Sequential(
            nn.Linear(expert_dim, task_dim),
            nn.ReLU(),
            nn.LayerNorm(task_dim),
            nn.Dropout(0.1),
            nn.Linear(task_dim, 1)
        )
    
    def forward(self, num_feats, cat_feats):
        # mlp
        num_out = self.num_linear(num_feats)
        
        # 处理类别特征
        cat_embeds = []
        for name, embed in self.cat_embeddings.items():
            cat_embeds.append(embed(cat_feats[name]))
        
        # 拼接所有特征
        cat_out = torch.cat(cat_embeds, dim=1)
        shared_features = torch.cat([num_out, cat_out], dim=1)
        
        # MMoE特征
        expert_features = [expert(shared_features) for expert in self.experts]
        expert_features = torch.stack(expert_features, dim=1) # [batch_size, num_experts, expert_dim]
        
        # gate
        ctr_gate = self.ctr_gate(shared_features).unsqueeze(2) # 第二个维度新增一维度 [batch_size, num_experts, 1]
        cvr_gate = self.cvr_gate(shared_features).unsqueeze(2)
        
        # 激活特征
        ctr_features = torch.sum(ctr_gate * expert_features, dim=1) # [batch_size, experts_dim]
        cvr_features = torch.sum(cvr_gate * expert_features, dim=1)
        
        # 分别计算 CTR 和 CVR
        ctr_logits = self.ctr_layer(ctr_features)
        cvr_logits = self.cvr_layer(cvr_features)
        
        return ctr_logits, cvr_logits
# << data preprocess >> 
import pandas
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict 

# 预处理
class Datapreprocess:
    def __init__(self, num_features=None, cat_features=None, random_seed=42):
        
        self.set_random_seed(random_seed)
        self.num_features = num_features or []
        self.cat_features = cat_features or []
        #self.seq_features = seq_features or []
        #self.max_seq_len = max_seq_len
        self.scaler = StandardScaler()
        # 类别vocabs
        self.cat_vocabs = defaultdict(dict)
    
    """拟合数据，学习预处理参数"""
    def fit(self, df):
        
        # 处理数值特征
        if self.num_features:
            self.scaler.fit(df[self.num_features])
        
        # 处理类别特征
        for feature in self.cat_features:
            unique_values = df[feature].unique()
            # 为每个类别分配一个唯一索引，从1开始（0保留给填充）
            self.cat_vocabs[feature] = {val: i+1 for i, val in enumerate(unique_values)}
        
        return self
    
    """设置随机种子确保结果可复现"""
    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    """转换数据，应用预处理参数"""
    def transform(self, df):
        
        processed_data = {}
        
        # 处理数值特征
        if self.num_features:
            num_data = self.scaler.transform(df[self.num_features])
            processed_data['num_features'] = torch.FloatTensor(num_data)
        
        # 处理类别特征
        if self.cat_features:
            cat_data = {}
            for feature in self.cat_features:
                # 将类别转换为索引，未知类别用0表示
                indices = df[feature].map(lambda x: self.cat_vocabs[feature].get(x, 0)).values
                cat_data[feature] = torch.LongTensor(indices)
            processed_data['cat_features'] = cat_data
        
        # item_id，保持原来的值，DIN的时候用，item和seq计算att
        # processed_data['item_id'] = torch.FloatTensor(df['item_id'].values)
        
        # 处理label
        processed_data['click_label'] = torch.FloatTensor(df['click_label'].values)
        processed_data['order_label'] = torch.FloatTensor(df['order_label'].values)
        
        return processed_data
    
#  序列预处理
class SeqDatapreprocess:
    def __init__(self, num_features=None, cat_features=None, seq_features=None, max_seq_len=20, random_seed=42):
        
        self.set_random_seed(random_seed)
        self.num_features = num_features or []
        self.cat_features = cat_features or []
        self.seq_features = seq_features or []
        self.max_seq_len = max_seq_len
        self.scaler = StandardScaler()

        # 类别vocabs
        self.cat_vocabs = defaultdict(dict)
    
    # str2list
    def parse_seq(self, seq_str):
        return [int(num) for num in seq_str[0].split(',')]

    # fit
    def fit(self, df):
        
        # 处理数值特征
        if self.num_features:
            self.scaler.fit(df[self.num_features])
        
        # 处理类别特征
        for feature in self.cat_features:
            unique_values = df[feature].unique()
            # 分配唯一索引
            self.cat_vocabs[feature] = {val: i+1 for i, val in enumerate(unique_values)}
        
        return self
    
    # seed
    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # transform
    def transform(self, df):
        
        processed_data = {}
        
        # 处理数值特征
        if self.num_features:
            num_data = self.scaler.transform(df[self.num_features])
            processed_data['num_features'] = torch.FloatTensor(num_data)
        
        # 处理类别特征
        if self.cat_features:
            cat_data = {}
            for feature in self.cat_features:
                # 将类别转换为索引
                indices = df[feature].map(lambda x: self.cat_vocabs[feature].get(x, 0)).values
                cat_data[feature] = torch.LongTensor(indices)
            processed_data['cat_features'] = cat_data
        
        # item_id，保持原来的值，DIN的时候用，item和seq计算att
        processed_data['item_id'] = torch.LongTensor(df['item_id'].values)

        # 处理序列特征
        seq_data = df[self.seq_features].values
        seq_array = [self.parse_seq(seq) for seq in seq_data]
        seq_array = np.array(seq_array, dtype=np.int64)
        seq_tensor = torch.from_numpy(seq_array).long()            
        processed_data['seq_id'] = seq_tensor
        
        # 处理label
        processed_data['click_label'] = torch.FloatTensor(df['click_label'].values)
        processed_data['order_label'] = torch.FloatTensor(df['order_label'].values)
        
        return processed_data
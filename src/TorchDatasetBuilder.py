import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
import logging

from aibrain_common.data.transform import (
    ToNumpy,
    ToSparseTensorValue
)
from aibrain_common.data.dataset import DataSet
from aibrain_common.io.hdfs_reader import HdfsReader
from aibrain_common.io.local_table_reader import CSVReader
from aibrain_common.io.pandas_reader import PandasReader
from aibrain_common.io.config.configs import *

from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


SCHEMA_DTYPE = {
    'int32': np.int32,
    'int64': np.uint64,
    'float': np.float32,
    'float32': np.float32,
    'float64': np.float64,
    'double': np.double,
    'string': np.str,
    'boolean': np.bool,
}

class ColumnSpec(object):
    def __init__(self,
                 column_name,
                 is_sparse=False,
                 is_categorical=False,
                 is_label=False,
                 shape=None,
                 tag=None,
                 separator=None,
                 group_separator=None,
                 dtype=None,
                 group_num=None):
        """
        table column spec
        Args:
            column_name: table column name
            is_sparse (boolean): If True, it is sparse feature which will be transformed to SparseTensorValue
            is_categorical(boolean): If True, it is categorical feature with will be useful in lightGBM training
            is_label (boolean): is label which will be added to labels for estimator input
            tag (string): feature extractor tag, same with column_name default
            separator (string): separator for dense feature, like `1.5,1.6,1.7` and separator is `,`
            dtype (string): dtype for dense feature, see SCHEMA_DTYPE for supported types
            group_name (int): feature extractor group number
        """
        self.name = column_name
        self.is_sparse_col = is_sparse
        self.is_label = is_label
        self.is_categorical_col = is_categorical
        self.shape = shape
        self.tag = tag
        self.separator = separator
        self.group_separator = group_separator
        self.dtype = dtype.lower() if dtype is not None else None
        self.group_num = group_num

    def is_sparse(self):
        return self.is_sparse_col

    def is_categorical(self):
        return self.is_categorical_col


class ColumnsMeta(object):
    def __init__(self,
                 name,
                 dtype,
                 type,
                 shape,
                 separator=None,
                 group_separator=None,
                 group=None,
                 feature_lib_tag=None,
                 is_label=False
                 ):
        """

        Parameters
        ----------
        name: feature column name
        dtype: data type, same as table schema
        type:  `dense` or `sparse`
        shape: list, like [342]
        separator: separator to split feature value
        group_separator: group separator for feature extractor
        group: group number
        feature_lib_tag: feature extractor tag
        is_label: boolean
        """
        self.name = name
        self.dtype = dtype
        self.type = type
        self.shape = shape
        self.separator = separator
        self.group_separator = group_separator
        self.group = group
        self.feature_lib_tag = feature_lib_tag
        self.is_label = is_label

    def get_property_dict(self):
        meta = {}
        for key, value in vars(self).items():
            if value is not None:
                meta[value] = value
        return meta


def get_shape_with_group_from_pandas(feature_map_df):
    """
    Args:
        feature_map_df: feature map data frame
        +-----------+--------------+---------------------+-------------+----------+
        |   id     |   feature    |   feature_crc64     | feature_type | group   |
        +----------+--------------+---------------------+------------------------+
        | 166605  |    166605    | 5000086947293856390 |  deep        |   1     |
        +----------+--------------+---------------------+------------------------+
        | 165163  |    165163    | 4900085252672674736 |  deep        |   1     |
        +----------+--------------+---------------------+------------------------+
        |   0     |      0      | 1000012592078815293 |  wide        |         |
        +----------+--------------+---------------------+------------------------+
        |   1     |      1      | 10000127582342479564|  wide        |         |
        +----------+--------------+---------------------+------------------------+
    Returns:
        shape from feature map data frame
    """
    feature_shape_df = feature_map_df.groupby(['feature_type', 'group'])['id'].count()
    feature_shape = feature_shape_df.to_dict()
    feature_shape_dict = {}
    for key in feature_shape.keys():
        feature_shape_dict.setdefault(key[0], [])
        if isinstance(key[1], float) or isinstance(key[1], int) or key[1].isdigit():
            feature_shape_dict[key[0]].insert(int(key[1]), feature_shape[key])
        else:
            feature_shape_dict[key[0]].insert(0, feature_shape[key])
    return feature_shape_dict


def get_shape_no_group_from_pandas(feature_map_df):
    feature_shape_df = feature_map_df.groupby('feature_type')['id'].count()
    feature_shape_df = feature_shape_df + 1
    feature_shape_dict = dict()
    feature_shape_df_dict = feature_shape_df.to_dict()
    for feature in feature_shape_df.to_dict():
        feature_shape_dict.setdefault(feature, [feature_shape_df_dict[feature]])
    return feature_shape_dict

# torch的逻辑，在读取的时候处理数据，供Dataloader调用
class TorchDatasetAdapter(Dataset):
    
    def __init__(self, dataset_builder):
        """
        Args:
            dataset_builder: 已初始化的 DatasetBuilder 实例（包含数据和特征配置）
        """
        self.builder = dataset_builder
        self.data = self._load_and_process_data()
    
    def _load_and_process_data(self):
        
        df = self.builder.to_pandas()
        
        # 处理每一条数据
        processed_data = []
        
        for _, row in df.iterrows():
            features = {}
            labels = {}
            
            for col_info in self.builder.columns.info:
                col_name = col_info.name
                value = row[col_name]
                if col_info.type == 'sparse':
                    # 处理稀疏特征（转换为 PyTorch 稀疏张量或稠密张量）
                    tensor = self._process_sparse_feature(value, col_info)
                    
                else:
                    # 处理密集特征（转换为 PyTorch 稠密张量）
                    tensor = self._process_dense_feature(value, col_info)
                
                # 区分特征和标签
                if col_info.is_label:
                    labels[col_name] = tensor
                else:
                    features[col_name] = tensor
            
            processed_data.append((features, labels))
        
        return processed_data
    
    def _process_sparse_feature(self, value, col_info):
        """处理稀疏特征：将字符串解析为 PyTorch 张量"""
        # 示例：假设稀疏特征格式为 "id1:val1\x01id2:val2"（单组）或多组用\x02分隔
        if col_info.group == 1:
            # 单组稀疏特征：解析为 (batch_size, shape) 的稠密张量（或稀疏张量）
            items = str(value).split(col_info.separator)  # 按分隔符拆分
            indices = []
            values = []
            for item in items:
                if not item:
                    continue
                try:
                    idx, val = item.split(':', 1)  # 拆分id和值
                    indices.append(int(idx))
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue  # 跳过格式错误的项
            
            # 初始化张量（根据特征形状填充）
            shape = col_info.shape[0] if col_info.shape else 1
            tensor = torch.zeros(shape, dtype=torch.float32)
            if indices:
                tensor[indices] = torch.tensor(values, dtype=torch.float32)
            return tensor
        else:
            # 多组稀疏特征：按组拆分后分别处理，返回张量列表或堆叠后的张量
            groups = str(value).split(col_info.group_separator)  # 按组分隔符拆分
            group_tensors = []
            for group in groups[:col_info.group]:  # 限制组数量
                items = str(group).split(col_info.separator)
                indices = []
                values = []
                for item in items:
                    if not item:
                        continue
                    try:
                        idx, val = item.split(':', 1)
                        indices.append(int(idx))
                        values.append(float(val))
                    except (ValueError, TypeError):
                        continue
                # 每组的形状
                group_shape = col_info.shape[len(group_tensors)] if col_info.shape else 1
                group_tensor = torch.zeros(group_shape, dtype=torch.float32)
                if indices:
                    group_tensor[indices] = torch.tensor(values, dtype=torch.float32)
                group_tensors.append(group_tensor)
            
            # 多组特征堆叠为一个张量（或保持列表）
            return torch.stack(group_tensors)

    def _process_dense_feature(self, value, col_info):
        """处理密集特征：将字符串解析为 PyTorch 稠密张量"""
        # 示例：假设密集特征格式为 "val1,val2,val3"（按分隔符拆分）
        if col_info.shape[0] == 1:
            # 标量密集特征（形状为1）
            try:
                return torch.tensor([float(value)], dtype=torch.float32)
            except (ValueError, TypeError):
                return torch.tensor([0.0], dtype=torch.float32)  # 异常值处理
        else:
            # 多维密集特征（按分隔符拆分后转换为张量）
            values = str(value).split(col_info.separator)
            # 转换为浮点数，不足补0，超出截断
            try:
                float_values = [float(v) for v in values if v]
            except (ValueError, TypeError):
                float_values = []
            
            # 对齐形状
            target_shape = col_info.shape[0]
            if len(float_values) < target_shape:
                float_values += [0.0] * (target_shape - len(float_values))
            else:
                float_values = float_values[:target_shape]
            
            return torch.tensor(float_values, dtype=torch.float32)

    def __getitem__(self, idx):
        """获取索引对应的样本（特征+标签）"""
        return self.data[idx]
    
    def __len__(self):
        """数据集大小"""
        return len(self.data)
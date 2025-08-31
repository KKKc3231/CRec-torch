import os
import sys
import yaml
import logging
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # /data/cc/GR_recommand/CRec_torch
sys.path.append(str(project_root))

from CRec_torch.scripts.logging import setup_logger

# logger = setup_logger(__name__, log_file='experiments/train.log')
# train_conf = yaml.load(open(os.path.join(project_root, 'option/train/train_esmm.yml')), Loader=yaml.FullLoader)
def register_model(train_conf, log_file):
    # set logger
    logger = setup_logger(__name__, log_file=log_file)
    # register
    model_name = train_conf['network']['type']
    logger.info(f"register model: {model_name}")

    # module
    module_name = f"ctr_model.{model_name.lower()}_arch"
    logger.info(f"module name: {module_name}")

    module = importlib.import_module(module_name)
    model_class = getattr(module, model_name)
    logger.info(f"model class: {model_class}")

    # 加载num 和 embedding
    feature_conf = train_conf['dataprocess']['features_conf_path']
    with open(feature_conf, 'r') as f:
        feature_conf = yaml.load(f, Loader=yaml.FullLoader)
        num_feature_dim = feature_conf['num_feature_dim']
        cat_feature_sizes = feature_conf['cat_feature_sizes']
    logger.info(f"加载特征信息成功～") 

    # 实例化模型
    if model_name == 'ESMM':
        embedding_dim = train_conf['network']['embedding_dim']
        mlp_dim = train_conf['network']['mlp_dim']
        task_dim = train_conf['network']['task_dim']
        logger.info(f"embedding_dim: {embedding_dim}, mlp_dim: {mlp_dim}, task_dim: {task_dim}")
        model = model_class(num_feature_dim, cat_feature_sizes, embedding_dim, mlp_dim, task_dim)
        logger.info(f"实例化ESMM模型成功～")

    # ESMM with MoE
    elif model_name == 'ESMMMOE':
        embedding_dim = train_conf['network']['embedding_dim']
        mlp_dim = train_conf['network']['mlp_dim']
        task_dim = train_conf['network']['task_dim']
        num_experts = train_conf['network']['num_experts']
        experts_dim = train_conf['network']['experts_dim']
        logger.info(f"embedding_dim: {embedding_dim}, mlp_dim: {mlp_dim}, task_dim: {task_dim}, num_experts: {num_experts}, experts_dim: {experts_dim}")
        model = model_class(num_feature_dim, cat_feature_sizes, embedding_dim, mlp_dim, task_dim, num_experts, experts_dim)
        logger.info(f"实例化ESMMMOE模型成功～")

    # 改进后的ESMM with MoE
    elif model_name == 'ENESMM':
        embedding_dim = train_conf['network']['embedding_dim']
        mlp_dim = train_conf['network']['mlp_dim']
        task_dim = train_conf['network']['task_dim']
        num_experts = train_conf['network']['num_experts']
        experts_dim = train_conf['network']['experts_dim']
        model = model_class(num_feature_dim, cat_feature_sizes, embedding_dim, mlp_dim, task_dim, num_experts, experts_dim)
        logger.info(f"实例化ENESMM模型成功～")

    # ESMM with DIN feature
    elif model_name == 'ESMMDIN':
        embedding_dim = train_conf['network']['embedding_dim']
        mlp_dim = train_conf['network']['mlp_dim']
        task_dim = train_conf['network']['task_dim']
        item_counts = train_conf['network']['item_counts']
        padding_idx = train_conf['network']['padding_idx']
        item_embedding_dim = train_conf['network']['item_embedding_dim']
        model = model_class(num_feature_dim, cat_feature_sizes, item_counts, padding_idx, item_embedding_dim, embedding_dim, mlp_dim, task_dim)
        logger.info(f"实例化ESMMDIN模型成功～")

    return model

def register_optimizer(train_conf, model, log_file):
    optimizer_name = train_conf['optimizer']['type']
    logger = setup_logger(__name__, log_file=log_file)

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=train_conf['optimizer']['lr'])
        logger.info(f"实例化{optimizer_name}优化器成功～")
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=train_conf['optimizer']['lr'])
        logger.info(f"实例化{optimizer_name}优化器成功～")
    else:
        raise ValueError(f"不支持的优化器类型：{optimizer_name}")

    return optimizer

def register_loss(train_conf, log_file):
    ctr_loss_name = train_conf['loss']['ctr_loss']
    ctcvr_loss_name = train_conf['loss']['ctcvr_loss']
    logger = setup_logger(__name__, log_file=log_file)
    
    # init loss
    if ctr_loss_name == 'BCEWithLogitsLoss':
        ctr_loss = nn.BCEWithLogitsLoss()
        logger.info(f"ctr_loss: 实例化{ctr_loss_name}损失函数成功～")
    
    if ctcvr_loss_name == 'BCEWithLogitsLoss':
        ctcvr_loss = nn.BCEWithLogitsLoss()
        logger.info(f"ctcvr_loss: 实例化{ctcvr_loss_name}损失函数成功～")

    return ctr_loss, ctcvr_loss   
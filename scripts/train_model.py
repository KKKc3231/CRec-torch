import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
project_root = Path(__file__).resolve().parent.parent  # /data/cc/GR_recommand/CRec_torch
sys.path.append(str(project_root))

from CRec_torch.scripts.logging import setup_logger
from CRec_torch.scripts.infer_model import infer_auc

# training pipline
def training_model(train_loader, val_loader, test_loader, model, optimizer, ctr_loss_f, ctcvr_loss_f, total_epochs, device, log_file, is_seq, model_save_path):
    logger = setup_logger('training', log_file)
    model.to(device)
    logger.info(f"开始训练模型～")
    model.train()

    # 开始训练
    for epoch in range(total_epochs):
        epoch_loss = 0.0
        logger.info(f"Epoch {epoch + 1}/{total_epochs}")
        total_step = len(train_loader)
        # 
        train_progress = tqdm(enumerate(train_loader), total=total_step, desc=f"训练 Epoch {epoch + 1}")
        for step, batch_data in train_progress:
            optimizer.zero_grad()
            num_feats = batch_data['num_features'].to(device)
            cat_feats = {k: v.to(device) for k, v in batch_data['cat_features'].items()}
            click_label = batch_data['click_label'].float().view(-1, 1).to(device) # ctr
            order_label = batch_data['order_label'].float().view(-1, 1).to(device) # cvr
            # 是否建模用户点击序列
            if is_seq:
                item_id = batch_data['item_id'].unsqueeze(1).to(device)
                seq_id = batch_data['seq_id'].to(device) 
                ctr_logits, cvr_logits = model(num_feats, cat_feats, item_id, seq_id)
            else:
                # forward
                ctr_logits, cvr_logits = model(num_feats, cat_feats)
            
            # esmm loss
            ctr_loss = F.binary_cross_entropy_with_logits(ctr_logits, click_label)
            ctcvr_loss = F.binary_cross_entropy(
                torch.sigmoid(ctr_logits) * torch.sigmoid(cvr_logits), 
                click_label * order_label
            )

            loss = ctr_loss + ctcvr_loss

            # backwaed
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if(step + 1) % 200 == 0:
                logger.info(f"Epoch {epoch + 1}, Step {step + 1}, ctr_loss: {ctr_loss.item()}, ctcvr_loss: {ctcvr_loss.item()}, total loss: {loss.item()}")

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")
        
        # infer auc on val dataset
        ctr_auc_val, cvr_auc_val = infer_auc(val_loader, model, ctr_loss_f, ctcvr_loss_f, device, log_file, is_seq)
        ctr_auc_test, cvr_auc_test = infer_auc(test_loader, model, ctr_loss_f, ctcvr_loss_f, device, log_file, is_seq)
        logger.info(f"Epoch {epoch + 1} ctr_auc_val: {ctr_auc_val:.4f}, cvr_auc_val: {cvr_auc_val:.4f}")
        logger.info(f"Epoch {epoch + 1} ctr_auc_test: {ctr_auc_test:.4f}, cvr_auc_test: {cvr_auc_test:.4f}")
    return model


import sys
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).resolve().parent.parent  # /data/cc/GR_recommand/CRec_torch
sys.path.append(str(project_root))

from CRec_torch.scripts.logging import setup_logger

# infer auc
def infer_auc(test_loader, model, ctr_loss_f, ctcvr_loss_f, device, log_file, is_seq=False):
    logger = setup_logger('training', log_file)
    model.eval()
    val_loss = 0.0
    val_batches = 0
    all_click_labels = []   
    all_click_probs = []
    all_order_labels = []
    all_order_probs = []
    
    # save best click_auc_model && best order_auc_model
    best_ctr_auc = 0.0
    best_cvr_auc = 0.0

    total_step = len(test_loader) 
    test_progress = tqdm(enumerate(test_loader), total=total_step, desc=f"infer auc on test dataset")
    with torch.no_grad():
        for step, batch_data in test_progress:
            # load val data
            num_feats = batch_data['num_features'].to(device)
            cat_feats = {k: v.to(device) for k, v in batch_data['cat_features'].items()}
            click_label = batch_data['click_label'].float().view(-1, 1).to(device)
            order_label = batch_data['order_label'].float().view(-1, 1).to(device)
            
            # 是否序列
            if is_seq:
                item_id = batch_data['item_id'].unsqueeze(1).to(device)
                seq_id = batch_data['seq_id'].to(device) 
                ctr_logits, cvr_logits = model(num_feats, cat_feats, item_id, seq_id)
            else:
                # forward
                ctr_logits, cvr_logits = model(num_feats, cat_feats)

            # loss
            ctr_loss = ctr_loss_f(ctr_logits, click_label)
            ctcvr_loss = F.binary_cross_entropy(
                torch.sigmoid(ctr_logits) * torch.sigmoid(cvr_logits), 
                click_label * order_label
            )

            loss = ctr_loss + ctcvr_loss
            val_loss += loss.item()
            val_batches += 1

            # 
            click_probs = torch.sigmoid(ctr_logits).cpu().numpy()
            order_probs = torch.sigmoid(cvr_logits).cpu().numpy()
            
            
            all_click_labels.extend(click_label.cpu().numpy())
            all_click_probs.extend(click_probs)
            all_order_labels.extend(order_label.cpu().numpy())
            all_order_probs.extend(order_probs)

        avg_val_loss = val_loss / val_batches
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        all_click_labels = np.array(all_click_labels).ravel()
        all_click_probs = np.array(all_click_probs).ravel()
        all_order_labels = np.array(all_order_labels).ravel()
        all_order_probs = np.array(all_order_probs).ravel()

        # ctr auc && cvr_auc
        cvr_auc = roc_auc_score(all_order_labels, all_order_probs)
        ctr_auc = roc_auc_score(all_click_labels, all_click_probs)

    return ctr_auc, cvr_auc
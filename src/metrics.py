import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# gauc

# ndcg@k

# hr@k

# auc
def infer_auc(model, TestDataLoader):
    # infer
    device='cuda'
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
    
    with torch.no_grad():
        for batch_data in TestDataLoader:
            # load val data
            num_feats = batch_data['num_features'].to(device)
            cat_feats = {k: v.to(device) for k, v in batch_data['cat_features'].items()}
            click_label = batch_data['click_label'].float().view(-1, 1).to(device)
            order_label = batch_data['order_label'].float().view(-1, 1).to(device)

            ctr_logits, cvr_logits = model(num_feats, cat_feats)

            # loss
            ctr_loss = F.binary_cross_entropy_with_logits(ctr_logits, click_label)
            #cvr_weights = click_label
            #cvr_loss = F.binary_cross_entropy_with_logits(cvr_logits, order_label, weight=cvr_weights)
            #cvr_loss = cvr_loss / (click_label.sum() + 1e-8)
            ctcvr_loss = F.binary_cross_entropy(
                torch.sigmoid(ctr_logits) * torch.sigmoid(cvr_logits), 
                order_label
            )

            loss = ctr_loss + ctcvr_loss
            val_loss += loss.item()
            val_batches += 1

            # auc
            click_probs = torch.sigmoid(ctr_logits).cpu().numpy()
            order_probs = torch.sigmoid(cvr_logits).cpu().numpy()

            all_click_labels.extend(click_label.cpu().numpy())
            all_click_probs.extend(click_probs)
            all_order_labels.extend(order_label.cpu().numpy())
            all_order_probs.extend(order_probs)

    avg_val_loss = val_loss / val_batches
    all_click_labels = np.array(all_click_labels).ravel()
    all_click_probs = np.array(all_click_probs).ravel()
    all_order_labels = np.array(all_order_labels).ravel()
    all_order_probs = np.array(all_order_probs).ravel()

    # CTR && CVR AUC
    cvr_auc = roc_auc_score(all_order_labels, all_order_probs)
    ctr_auc = roc_auc_score(all_click_labels, all_click_probs)
    
    return ctr_auc, cvr_auc
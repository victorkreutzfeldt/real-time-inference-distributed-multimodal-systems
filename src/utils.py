# utils.py

import torch
import numpy as np
from collections import Counter

import ast

import random 

# ========================== Random Seed ==========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass

# ========================== Accuracy Functions ==========================
def subset_accuracy_from_label_lists(true_label_lists, predicted_label_lists):
    """Exact match accuracy; 1.0 if predicted set equals true set, else 0"""
    acc_list = []
    for true_labels, pred_labels in zip(true_label_lists, predicted_label_lists):
        acc_list.append(set(true_labels) == set(pred_labels))
    accs = sum(acc_list) / len(acc_list)
    return accs

def hamming_accuracy_from_label_lists(true_label_lists, predicted_label_lists):
    """Average per-example intersection over union of predicted and true label sets"""
    acc_list = []
    for true_labels, pred_labels in zip(true_label_lists, predicted_label_lists):
        set_true = set(true_labels)
        set_pred = set(pred_labels)
        if len(set_true) == 0 and len(set_pred) == 0:
            acc = 1.0  # perfect if both empty
        else:
            intersection = set_true.intersection(set_pred)
            union = set_true.union(set_pred)
            acc = len(intersection) / len(union) if len(union) > 0 else 0.0
        acc_list.append(acc)
    accs = sum(acc_list) / len(acc_list)
    return accs


def compute_pos_weight(dataset, num_classes=29, device='cpu'):
    """"""
    # Series of lists or string representations
    all_labels_lists = dataset.annotations['label_idx'] 
    
    # Ensure correct interpretation of labels lists
    def safe_literal_eval(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x
    
    all_labels = [safe_literal_eval(lbls) for lbls in all_labels_lists]
    
    # Total number of samples
    N = len(all_labels)
    
    # For each class, count number of samples where class label is present
    positive_counts = np.zeros(num_classes, dtype=np.float32)
    for labels in all_labels:
        for label in set(labels):  # unique labels per sample (avoid double counting)
            positive_counts[label] += 1
    
    # Avoid division by zero
    positive_counts = np.clip(positive_counts, a_min=1, a_max=None)
    
    negative_counts = N - positive_counts
    
    pos_weight = negative_counts / positive_counts  # shape (num_classes,)
    
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float, device=device)

    return pos_weight_tensor

# # ========================== Evaluation ==========================
# def evaluate_per_chunk(model, dataloader, criterion, logger, device, modality='multimodal'):
#     model.eval()
#     total_loss = 0.0
#     total_hamming_acc = 0.0
#     total_subset_acc = 0.0
#     total_batches = 0

#     with torch.no_grad():
#         for batch in dataloader:
#             audio_feats = batch["audio"].to(DEVICE)
#             video_feats = batch["video"].to(DEVICE)
#             labels = batch["label"].to(DEVICE)

#             outputs = model(audio_feats, video_feats)  # Expected logits shape: (batch_size, num_classes)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()

#             probas = torch.sigmoid(outputs)
#             preds = (probas > 0.5).float()

#             predicted_labels = [torch.nonzero(row).squeeze(1).tolist() for row in preds]
#             true_label_indices = [torch.nonzero(row).squeeze(1).tolist() for row in labels]

#             # Hamming accuracy for single-label can be computed similarly
#             hamming_acc = hamming_accuracy_from_label_lists(true_label_indices, predicted_labels)
#             subset_acc = subset_accuracy(true_label_indices, predicted_labels)

#             total_hamming_acc += hamming_acc
#             total_subset_acc += subset_acc
#             total_batches += 1

#     avg_loss = total_loss / total_batches
#     avg_hamming_acc = total_hamming_acc / total_batches
#     avg_subset_acc = total_subset_acc / total_batches

#     logger.info(f"Evaluation - Avg Loss: {avg_loss:.4f} | Hamming Acc: {avg_hamming_acc:.4f} | Subset Acc: {avg_subset_acc:.4f}")
#     return avg_loss, avg_hamming_acc, avg_subset_acc

# # ========================== Training ==========================
# def train_per_chunk(model, criterion, optimizer, train_loader, val_loader, num_epochs=50):
#     best_val_hamming_acc = 0.0
#     patience_counter = 0

#     train_losses, val_losses = [], []
#     train_hamming_accs, val_hamming_accs = [], []
#     train_subset_accs, val_subset_accs = [], []

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss, running_hamming_acc, running_subset_acc = 0.0, 0.0, 0.0
#         batches = 0

#         logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
#         for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", ascii=True):
#             audio = batch["audio"].to(DEVICE)
#             video = batch["video"].to(DEVICE)
#             labels = batch["label"].to(DEVICE)

#             optimizer.zero_grad()
#             outputs = model(audio, video)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             probas = torch.sigmoid(outputs)
#             preds = (probas > 0.5).float()

#             predicted_labels = [torch.nonzero(row).squeeze(1).tolist() for row in preds]
#             true_label_indices = [torch.nonzero(row).squeeze(1).tolist() for row in labels]
            
#             hamming_acc = hamming_accuracy_from_label_lists(true_label_indices, predicted_labels)
#             subset_acc = subset_accuracy(true_label_indices, predicted_labels)

#             running_loss += loss.item()
#             running_hamming_acc += hamming_acc
#             running_subset_acc += subset_acc
#             batches += 1

#         train_loss = running_loss / batches
#         train_hamming_acc = running_hamming_acc / batches
#         train_subset_acc = running_subset_acc / batches

#         val_loss, val_hamming_acc, val_subset_acc = evaluate(model, val_loader, criterion)

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         train_hamming_accs.append(train_hamming_acc)
#         val_hamming_accs.append(val_hamming_acc)
#         train_subset_accs.append(train_subset_acc)
#         val_subset_accs.append(val_subset_acc)

#         logger.info(f"Epoch {epoch+1} summary - Train Loss: {train_loss:.4f} | Train Hamming Acc: {train_hamming_acc:.4f} | Train Subset Acc: {train_subset_acc:.4f} | Val Loss: {val_loss:.4f} | Val Hamming Acc: {val_hamming_acc:.4f} | Val Subset Acc: {val_subset_acc:.4f}")

#         if val_hamming_acc > best_val_hamming_acc:
#             best_val_hamming_acc = val_hamming_acc
#             patience_counter = 0
#             state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
#             torch.save(state_dict, OUTPUT_MODEL_PATH)
#             logger.info(f"✅ New best model saved at epoch {epoch+1} with Val Hamming Acc: {val_hamming_acc:.4f}")
#         else:
#             patience_counter += 1
#             if patience_counter >= PATIENCE:
#                 logger.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
#                 break

#     return train_losses, train_hamming_accs, train_subset_accs, val_losses, val_hamming_accs, val_subset_accs
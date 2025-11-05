# src/utils.py

import torch
import numpy as np
from collections import Counter

import ast

import random 


# ---- Random Seed ----
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

# ---- Accuracy Functions ----
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
    """Compute positive weights for each class based on dataset annotations."""
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
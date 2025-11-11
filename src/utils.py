# src/utils.py

"""
Utility functions for reproducibility, multilabel accuracy metrics, positional weighting, 
and fallback embedding extraction for audio pipelines.

This module provides:
    - Seed setting for consistent experiment results across random modules and devices.
    - Exact subset and Hamming accuracy calculations for multilabel classification.
    - Computation of positive class weights to handle class imbalance.
    - Extraction of fallback audio embeddings from silent audio used in robustness scenarios.

@author Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
@date 2025-11-11
"""

import torch
import numpy as np

import ast

import random 

from src.vggish_input import waveform_to_examples


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value to set. Default is 42.
    """
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


def subset_accuracy_from_label_lists(true_label_lists, predicted_label_lists) -> float:
    """
    Computes exact subset match accuracy between true and predicted multilabel sets.

    Args:
        true_label_lists (List[List[int]]): List of ground-truth label index lists per example.
        predicted_label_lists (List[List[int]]): List of predicted label index lists per example.

    Returns:
        float: Fraction of examples where predicted set exactly matches the true set.
    """
    acc_list = []
    for true_labels, pred_labels in zip(true_label_lists, predicted_label_lists):
        acc_list.append(set(true_labels) == set(pred_labels))
    accs = sum(acc_list) / len(acc_list)
    return accs


def hamming_accuracy_from_label_lists(true_label_lists, predicted_label_lists) -> float:
    """
    Computes average Hamming accuracy across multilabel examples.

    Defined as the average intersection-over-union (IoU) of predicted and true label sets per example.

    Args:
        true_label_lists (List[List[int]]): Ground-truth label indices per example.
        predicted_label_lists (List[List[int]]): Predicted label indices per example.

    Returns:
        float: Mean IoU accuracy score over all examples.
    """
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


def compute_pos_weight(dataset, num_classes: int = 29, device: str = 'cpu') -> torch.Tensor:
    """
    Compute positive class weights to address class imbalance in multilabel classification.

    Args:
        dataset: Dataset object containing token-level annotations in 'label_idx' column.
        num_classes (int): Number of label classes. Default 29.
        device (str or torch.device): Device to place the resulting tensor.

    Returns:
        torch.Tensor: Tensor of shape (num_classes,) with positive weights per class.
    """
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


def extract_fallback_audio_token_emb(pipeline, sample_rate=16000, token_duration=1.0, device='cpu'):
    """
    Extract a fallback audio token embedding from silent audio for fallback use.

    Args:
        pipeline (torch.nn.Module): Audio feature extraction pipeline.
        sample_rate (int, optional): Audio sample rate in Hz. Defaults to 16000.
        token_duration (float, optional): Token duration in seconds. Defaults to 1.0.
        device (torch.device or str, optional): Device for tensor operations. Defaults to 'cpu'.

    Returns:
        numpy.ndarray: Fallback embedding vector of shape consistent with pipeline output.
    """
    
    # Calculate expected length
    expected_len = int(token_duration * sample_rate)

    # Generate a silent waveform
    waveform = np.zeros(expected_len)

    # Extract mel-spectrogram example from this silent waveform
    spectrogram = waveform_to_examples(data=waveform, sample_rate=sample_rate, return_tensor=False)

    # Convert to tensor and send to device
    spectrogram = torch.tensor(spectrogram, device=device, dtype=torch.float32)
    
    # Generate fallback embedding from silent audio over a token
    with torch.no_grad():
        emb = pipeline(spectrogram.unsqueeze(1), return_embs=True)['embs']

    # Convert to numpy
    emb = emb.cpu().numpy().astype(np.float32).squeeze()

    return emb
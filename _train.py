#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import datetime
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from src.datasets import PerVideoMultimodalDataset
from src.models import (
    PerVideoBiLSTMAudioClassifier,
    PerVideoBiLSTMVideoClassifier,
    PerVideoBiLSTMMultimodalClassifier,
    PerVideoBiLSTMAGVisualAttnMultimodalClassifier
)
from src.utils import (
    set_seed,
    hamming_accuracy_from_label_lists,
    subset_accuracy_from_label_lists,
    compute_pos_weight
)


# ---- Argument Parsing ----
def parse_args():
    parser = argparse.ArgumentParser(description="Unified Audio/Video/Multimodal Per-Video Training")
    parser.add_argument("--modality", type=str, required=True, choices=["audio", "video", "multimodal"])
    parser.add_argument("--AGVattn", action="store_true", default=False,
                        help="Use Audio-Guided Visual Attention model (only valid with multimodal).")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9, help="Used only for SGD.")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pos-weight", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=False)

    # Model architecture style
    parser.add_argument("--style", type=str, default="base",
                        choices=["base", "reg", "deep", "wide", "res"],
                        help="Head architecture style for audio/video: base|reg|deep|wide|res.")

    # Attention
    parser.add_argument("--debug-attn", action="store_true", default=False,
                        help="Enable debug logging for audio-guided visual attention in multimodal model.")
    parser.add_argument("--attn-temperature", type=float, default=1,
                        help="Temperature for computing attention scores.")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Froze attention weights during warmup epochs.")
    parser.add_argument("--no-anneal-temperature", action="store_true", default=True,
                        help="If set, disables temperature annealing. Keeps attention temperature fixed at attn_temperature.")

    # Scheduler
    parser.add_argument("--use-scheduler", action="store_true", default=False,
                        help="Use learning rate scheduler (StepLR).")

    parser.add_argument("--scheduler-step-size", type=int, default=20,
                        help="StepLR step size (number of epochs between LR decay).")

    parser.add_argument("--scheduler-gamma", type=float, default=0.1,
                        help="StepLR gamma (multiplicative LR decay factor).")

    return parser.parse_args()


# ---- Device Setup ----
def setup_device_and_workers(args):
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if getattr(torch.backends, 'mps', None)
        and torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if args.num_workers == 0:
            try:
                import multiprocessing as mp
                args.num_workers = max(2, min(8, mp.cpu_count() // 2))
            except Exception:
                args.num_workers = 2
        args.pin_memory = True

    print(f"[INFO] Using device: {device}")
    return device, args


# ---- Logger Setup ----
def setup_logger(base_name):
    log_dir = "models/classification/per_video/logs/"
    os.makedirs(log_dir, exist_ok=True)
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"{base_name}_{current_time_str}.log")

    logger = logging.getLogger(base_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[ %(levelname)s : %(asctime)s ] - %(message)s")
    fh = logging.FileHandler(log_file_path, mode="w")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    return logger


# ---- Model Builder ----
def build_model(args, device):
    posw_suffix = "_posweight" if args.pos_weight else ""
    style_suffix = args.style

    if args.modality == "audio":
        base_name = f"classifier_audio_features_{style_suffix}{posw_suffix}"
        model_class = PerVideoBiLSTMAudioClassifier
        audio_path = "data/classification/embeddings/audio.h5"
        video_path = "data/classification/features/video_tem_spa_gpa.h5"
    elif args.modality == "video":
        base_name = f"classifier_video_features_{style_suffix}{posw_suffix}"
        model_class = PerVideoBiLSTMVideoClassifier
        audio_path = "data/classification/embeddings/audio.h5"
        video_path = "data/classification/features/video_tem_spa_gpa.h5"
    elif args.modality == "multimodal":
        if args.AGVattn:
            base_name = f"AGVattn_classifier_multimodal_features_{style_suffix}{posw_suffix}"
            model_class = PerVideoBiLSTMAGVisualAttnMultimodalClassifier
            audio_path = "data/classification/embeddings/audio.h5"
            video_path = "data/classification/features/video_tem_gpa.h5"
            warmup_epochs = args.warmup_epochs
        else:
            base_name = f"shallow_classifier_multimodal_features_{style_suffix}{posw_suffix}"
            model_class = PerVideoBiLSTMMultimodalClassifier
            audio_path = "data/classification/embeddings/audio.h5"
            video_path = "data/classification/features/video_tem_spa_gpa.h5"
            warmup_epochs = 0
    else:
        raise ValueError(f"Unsupported modality: {args.modality}")

    model = model_class(num_classes=29, style=args.style)
    if args.modality == "multimodal" and args.AGVattn:
        model = model_class(num_classes=29, style=args.style, debug=args.debug_attn, temperature=args.attn_temperature)

    model = model.to(device)
    return model, base_name, audio_path, video_path, warmup_epochs if 'warmup_epochs' in locals() else 0


# ---- Data Loaders Builder ----
def get_dataloaders(args, audio_path, video_path):
    dataset_kwargs = dict(
        annotations_file="data/annotations.csv",
        audio_h5_path=audio_path,
        video_h5_path=video_path,
        modality=args.modality
    )

    train_ds = PerVideoMultimodalDataset(split="train", **dataset_kwargs)
    val_ds = PerVideoMultimodalDataset(split="val", **dataset_kwargs)
    test_ds = PerVideoMultimodalDataset(split="test", **dataset_kwargs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.pin_memory)

    return train_loader, val_loader, test_loader


# ---- Other Utilities ----
def calculate_attention_entropy(attn_weights):
    epsilon = 1e-8
    attn = attn_weights + epsilon
    entropy = -(attn * attn.log()).sum(dim=1)
    return entropy.mean().item()


def set_attention_trainable(model, trainable: bool, logger):
    if hasattr(model, 'agva_attn'):
        for param in model.agva_attn.parameters():
            param.requires_grad = trainable
        logger.info(f"{'🔥 Unfroze' if trainable else '❄️ Froze'} attention parameters.")


def get_annealed_temperature(epoch, max_epoch, start_temp=1.0, end_temp=0.1):
    if epoch >= max_epoch:
        return end_temp
    return start_temp - (start_temp - end_temp) * (epoch / max_epoch)


def load_shallow_classifier_weights(attn_model, shallow_model_path, device, logger):
    if not os.path.isfile(shallow_model_path):
        logger.warning(f"Shallow model weights file not found: {shallow_model_path}")
        return

    shallow_ckpt = torch.load(shallow_model_path, map_location=device)
    shallow_state = shallow_ckpt  # adapt if checkpoint format
    attn_state = attn_model.state_dict()
    classifier_keys = [k for k in attn_state if k.startswith('classifier.')]
    filtered_shallow = {k: v for k, v in shallow_state.items() if k in classifier_keys}
    attn_state.update(filtered_shallow)
    attn_model.load_state_dict(attn_state)
    logger.info(f"Loaded shallow classifier weights from {shallow_model_path} into AG attention model.")


# ---- Training, Evaluation, and Plotting ----
@torch.no_grad()
def evaluate(model, dataloader, criterion, args, device, logger):
    model.eval()
    total_loss, total_h_acc, total_s_acc = 0.0, 0.0, 0.0

    if args.AGVattn and args.debug_attn:
        attn_entropy_sum = 0.0
        attn_max_sum = 0.0
        attn_min_sum = 0.0
        attn_batches = 0

    for batch in dataloader:
        if args.modality == "audio":
            X = batch["audio"].to(device)
            outputs = model(X)
        elif args.modality == "video":
            X = batch["video"].to(device).view(batch["video"].size(0), -1)
            outputs = model(X)
        else:
            X_audio = batch["audio"]
            X_video = batch["video"]
            if args.AGVattn and args.debug_attn:
                outputs, attn_weights, _ = model(X_audio.to(device), X_video.to(device))
                attn_entropy_sum += calculate_attention_entropy(attn_weights)
                attn_max_sum += attn_weights.max().item()
                attn_min_sum += attn_weights.min().item()
                attn_batches += 1
            else:
                outputs = model(X_audio.to(device), X_video.to(device))

        labels = batch["labels"].to(device)
        loss = criterion(outputs, labels)

        probas = torch.sigmoid(outputs)
        preds = (probas > 0.5).float()

        preds_reshaped = preds.view(-1, 29)
        labels_reshaped = labels.view(-1, 29)

        pred_labels = [torch.nonzero(row).squeeze(1).tolist() for row in preds_reshaped]
        true_labels = [torch.nonzero(row).squeeze(1).tolist() for row in labels_reshaped]

        total_loss += loss.item()
        total_h_acc += hamming_accuracy_from_label_lists(true_labels, pred_labels)
        total_s_acc += subset_accuracy_from_label_lists(true_labels, pred_labels)

    if args.AGVattn and args.debug_attn and attn_batches > 0:
        avg_entropy = attn_entropy_sum / attn_batches
        avg_max = attn_max_sum / attn_batches
        avg_min = attn_min_sum / attn_batches
        logger.info(f"[DEBUG] Eval avg attention entropy: {avg_entropy:.4f}, max: {avg_max:.4f}, min: {avg_min:.4f}")

    n = len(dataloader)
    return total_loss / n, total_h_acc / n, total_s_acc / n


def train(model, dataloader, criterion, optimizer, args, device, logger):
    model.train()
    total_loss, total_h_acc, total_s_acc = 0.0, 0.0, 0.0

    if args.AGVattn and args.debug_attn:
        attn_entropy_sum = 0.0
        attn_max_sum = 0.0
        attn_min_sum = 0.0
        attn_batches = 0

    for batch in tqdm(dataloader, desc="Training", ascii=True):
        if args.modality == "audio":
            X = batch["audio"].to(device)
            outputs = model(X)
        elif args.modality == "video":
            X = batch["video"].to(device)
            outputs = model(X)
        else:
            if args.AGVattn and args.debug_attn:
                outputs, attn_weights, _ = model(batch["audio"].to(device), batch["video"].to(device))
                attn_entropy_sum += calculate_attention_entropy(attn_weights)
                attn_max_sum += attn_weights.max().item()
                attn_min_sum += attn_weights.min().item()
                attn_batches += 1
            else:
                outputs = model(batch["audio"].to(device), batch["video"].to(device))

        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        probas = torch.sigmoid(outputs)
        preds = (probas > 0.5).float()

        preds_reshaped = preds.view(-1, 29)
        labels_reshaped = labels.view(-1, 29)

        pred_labels = [torch.nonzero(row).squeeze(1).tolist() for row in preds_reshaped]
        true_labels = [torch.nonzero(row).squeeze(1).tolist() for row in labels_reshaped]

        total_loss += loss.item()
        total_h_acc += hamming_accuracy_from_label_lists(true_labels, pred_labels)
        total_s_acc += subset_accuracy_from_label_lists(true_labels, pred_labels)

    if args.AGVattn and args.debug_attn and attn_batches > 0:
        avg_entropy = attn_entropy_sum / attn_batches
        avg_max = attn_max_sum / attn_batches
        avg_min = attn_min_sum / attn_batches
        logger.info(f"[DEBUG] Avg attention entropy: {avg_entropy:.4f}, max: {avg_max:.4f}, min: {avg_min:.4f}")

    n = len(dataloader)
    return total_loss / n, total_h_acc / n, total_s_acc / n


def train_model(model, train_loader, val_loader, criterion, optimizer, args, device, logger, base_name, warmup_epochs):
    best_val_h_acc = 0.0
    patience_counter = 0
    hist = {
        'train_loss': [], 'val_loss': [],
        'train_h_acc': [], 'val_h_acc': [],
        'train_s_acc': [], 'val_s_acc': []
    }

    if args.modality == "multimodal" and args.AGVattn:
        max_anneal_epochs = 20
        anneal_start = warmup_epochs
        anneal_end = warmup_epochs + max_anneal_epochs
        set_attention_trainable(model, False, logger)

    for epoch in range(args.epochs):
        logger.info(f"= Epoch {epoch+1}/{args.epochs} =")

        if args.modality == "multimodal" and args.AGVattn:
            if epoch == warmup_epochs:
                set_attention_trainable(model, True, logger)

            if args.no_anneal_temperature:
                new_temp = args.attn_temperature
            else:
                if epoch < anneal_start:
                    new_temp = args.attn_temperature
                elif epoch < anneal_end:
                    progress = epoch - anneal_start
                    new_temp = get_annealed_temperature(
                        progress, max_anneal_epochs,
                        start_temp=args.attn_temperature, end_temp=0.1
                    )
                else:
                    new_temp = 0.1

            model.agva_attn.set_temperature(new_temp)
            logger.info(f"🌡️ Epoch {epoch+1} attention temperature: {new_temp:.4f}")

        train_loss, train_h_acc, train_s_acc = train(model, train_loader, criterion, optimizer, args, device, logger)
        val_loss, val_h_acc, val_s_acc = evaluate(model, val_loader, criterion, args, device, logger)

        logger.info(f"[Train] Loss={train_loss:.4f} | HamAcc={train_h_acc:.4f} | SubAcc={train_s_acc:.4f}")
        logger.info(f"[Val]   Loss={val_loss:.4f} | HamAcc={val_h_acc:.4f} | SubAcc={val_s_acc:.4f}")

        hist['train_loss'].append(train_loss)
        hist['val_loss'].append(val_loss)
        hist['train_h_acc'].append(train_h_acc)
        hist['val_h_acc'].append(val_h_acc)
        hist['train_s_acc'].append(train_s_acc)
        hist['val_s_acc'].append(val_s_acc)

        if args.use_scheduler:
            optimizer.param_groups  # keep to access LR info if needed, but actual scheduler step below
            optimizer_scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
            optimizer_scheduler.step()  # Advance scheduler step
            logger.info(f"Learning rate after epoch {epoch+1}: {optimizer_scheduler.get_last_lr()}")

        if val_h_acc > best_val_h_acc:
            best_val_h_acc = val_h_acc
            patience_counter = 0
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, 
                       os.path.join("models/classification/per_video/", f"{base_name}.pth"))
            logger.info(f"✅ Saved new best model with Val HamAcc={val_h_acc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"⚠️ No validation improvement @ PATIENCE ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                logger.info("🛑 Early stopping")
                break

    return hist


def save_training_plots(history, base_name, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Losses - {base_name}')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    loss_plot_path = os.path.join(results_dir, f"{base_name}_losses.png")
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history['train_h_acc'], label='Train Hamming Acc', linewidth=2)
    plt.plot(history['val_h_acc'], label='Val Hamming Acc', linewidth=2)
    plt.plot(history['train_s_acc'], label='Train Subset Acc', linewidth=2)
    plt.plot(history['val_s_acc'], label='Val Subset Acc', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracies - {base_name}')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    acc_plot_path = os.path.join(results_dir, f"{base_name}_accuracies.png")
    plt.tight_layout()
    plt.savefig(acc_plot_path, dpi=150)
    plt.close()

    return loss_plot_path, acc_plot_path


# ---- Main program flow ----
def main():
    args = parse_args()
    set_seed(args.seed)
    device, args = setup_device_and_workers(args)

    model, base_name, audio_path, video_path, warmup_epochs = build_model(args, device)
    train_loader, val_loader, test_loader = get_dataloaders(args, audio_path, video_path)

    logger = setup_logger(base_name)

    logger.info("= Arguments =")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    logger.info("=====")
    logger.info(f"Modality: {args.modality} | Device: {device}")
    logger.info(f"Audio features path: {audio_path}")
    logger.info(f"Video features path: {video_path}")

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(model)
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")

    if args.pos_weight:
        pos_weight = compute_pos_weight(train_loader.dataset, num_classes=29, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer setup
    if args.modality == "multimodal" and args.AGVattn:
        attn_params = list(model.agva_attn.parameters())
        classifier_params = [p for n, p in model.named_parameters() if not n.startswith('agva_attn.')]

        lr_classifier = args.lr
        lr_attn = args.lr * 0.1
        optimizer = optim.AdamW([
            {'params': classifier_params, 'lr': lr_classifier},
            {'params': attn_params, 'lr': lr_attn}
        ], weight_decay=args.weight_decay)
        logger.info(f"Using AdamW optimizer with classifier lr={lr_classifier}, attention lr={lr_attn}")

    else:
        if args.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
                nesterov=True
            )
        elif args.optimizer == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(args.adam_beta1, args.adam_beta2)
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(args.adam_beta1, args.adam_beta2)
            )
        logger.info(f"Using optimizer: {optimizer}")

    # Scheduler
    if args.use_scheduler:
        scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        logger.info(f"Using StepLR scheduler with step_size={args.scheduler_step_size}, gamma={args.scheduler_gamma}")
    else:
        scheduler = None

    if args.train:
        history = train_model(model, train_loader, val_loader, criterion, optimizer, args, device, logger, base_name, warmup_epochs)
        loss_path, acc_path = save_training_plots(history, base_name, "models/classification/per_video/figs")
        logger.info(f"Saved training loss plot to {loss_path}")
        logger.info(f"Saved training accuracy plot to {acc_path}")
    else:
        logger.info("Skipping training...")

    # Load pre-trained model weights if available
    model_path = os.path.join("models/classification/per_video/", f"{base_name}.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")

    # Evaluate on test set
    test_loss, test_h_acc, test_s_acc = evaluate(model, test_loader, criterion, args, device, logger)
    logger.info(f"[Test] Loss={test_loss:.4f} | HamAcc={test_h_acc:.4f} | SubAcc={test_s_acc:.4f}")

    # Save results summary to CSV
    results_csv = "models/classification/per_video/test_results.csv"
    new_row = {
        "base_name": base_name,
        "modality": args.modality,
        "style": args.style,
        "optimizer": args.optimizer,
        "lr": args.lr if not args.AGVattn else [lr_classifier, lr_attn],
        "weight_decay": args.weight_decay,
        "test_loss": round(test_loss, 6),
        "test_hacc": round(test_h_acc, 6),
        "test_sacc": round(test_s_acc, 6),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_params": int(num_params),
        "num_trainable": int(num_trainable),
        "pos_weight": args.pos_weight
    }

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_results = pd.DataFrame([new_row])

    df_results.to_csv(results_csv, index=False)
    logger.info(f"Updated results table saved to {results_csv}")
    logger.info(f"\n{df_results.to_string(index=False)}")


if __name__ == "__main__":
    main()
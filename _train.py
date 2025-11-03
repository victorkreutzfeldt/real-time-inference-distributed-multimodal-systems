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

from src.per_video_models import (
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

# == Parser ==
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

# Standardization
parser.add_argument(
    "--std",
    type=str,
    choices=["none", "zscore", "minmax"],
    default="none",
    help="Form of data standardization to use: none, zscore, or minmax (default: minmax)."
)

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

args = parser.parse_args()

# Set seed
set_seed(args.seed)

# == Device Selection ==
DEVICE = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps') if getattr(torch.backends, 'mps', None)
    and torch.backends.mps.is_available() else
    torch.device('cpu')
)
USE_CUDA = (DEVICE.type == 'cuda')
if USE_CUDA:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.num_workers == 0:
        pass
        # try:
        #     import multiprocessing as mp
        #     args.num_workers = max(2, min(8, mp.cpu_count() // 2))
        # except Exception:
        #     args.num_workers = 2
    args.pin_memory = True
print(f"[INFO] Using device: {DEVICE}")

# == Config ==
ANNOTATIONS_PATH = "data/annotations.csv"

# Build standardized paths depending on user choice
if args.std == "none":
    AUDIO_PATH = "data/classification/embeddings/audio.h5"
    VIDEO_PATH = "data/classification/features/video_tem_spa_gpa.h5"
elif args.std == "zscore":
    AUDIO_PATH = "data/classification/embeddings/normalized/audio_zscore.h5"
    VIDEO_PATH = "data/classification/features/normalizedvideo_tem_spa_gpa_zscore.h5"
else: # minmax
    AUDIO_PATH = "data/classification/embeddings/normalized/audio_minmax.h5"
    VIDEO_PATH = "data/classification/features/normalized/video_tem_spa_gpa_minmax.h5"

NUM_CLASSES = 29

# Build a suffix capturing style and pos_weight usage
style_suffix = args.style  # 'base' | 'reg' | 'deep' | 'wide' | 'res'
posw_suffix = "_posweight" if args.pos_weight else ""

if args.modality == "audio":
    BASE_NAME = f"classifier_audio_features_{style_suffix}{posw_suffix}"
    ModelClass = PerVideoBiLSTMAudioClassifier

elif args.modality == "video":
    BASE_NAME = f"classifier_video_features_{style_suffix}{posw_suffix}"
    ModelClass = PerVideoBiLSTMVideoClassifier

elif args.modality == "multimodal":
    
    # AG attention (multimodal) uses non-spatial-GPA video features
    if args.AGVattn:
        BASE_NAME = f"AGVattn_classifier_multimodal_features_{style_suffix}{posw_suffix}"
        ModelClass = PerVideoBiLSTMAGVisualAttnMultimodalClassifier
        
        if args.std == "none": 
            VIDEO_PATH = "data/classification/features/video_tem_gpa.h5" 
        elif args.std == "zscore":
            VIDEO_PATH = "data/classification/features/video_tem_gpa_zscore_clipped.h5"
        else:
            VIDEO_PATH = "data/classification/features/video_tem_gpa_minmax_scaled.h5"

        WARMUP_EPOCHS = args.warmup_epochs # Number of epochs to freeze attention
    else:
        BASE_NAME = f"shallow_classifier_multimodal_features_{style_suffix}{posw_suffix}"
        ModelClass = PerVideoBiLSTMMultimodalClassifier

# DIRs
OUTPUT_MODEL_DIR = "models/classification/per_video/"
OUTPUT_FIGS_DIR = "models/classification/per_video/figs"
OUTPUT_CSV_DIR = "models/classification/per_video"
LOG_FILE_DIR = "models/classification/per_video/logs/"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGS_DIR, exist_ok=True)
os.makedirs(LOG_FILE_DIR, exist_ok=True)

# Paths
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_MODEL_DIR, f"{BASE_NAME}.pth")
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_FIGS_DIR, f"figs/{BASE_NAME}.png")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_CSV_DIR, "test_results.csv")

# Logging
current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, f"{BASE_NAME}_{current_time_str}.log")
logger = logging.getLogger(BASE_NAME)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[ %(levelname)s : %(asctime)s ] - %(message)s")
fh = logging.FileHandler(LOG_FILE_PATH, mode="w")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())

# = Log args used =
logger.info("= Arguments =")
for k, v in vars(args).items():
    logger.info(f"{k}: {v}")
logger.info("=====")
logger.info(f"Modality: {args.modality} | Device: {DEVICE}")
logger.info(f"Standardization form: {args.std}")
logger.info(f"Audio features path: {AUDIO_PATH}")
logger.info(f"Video features path: {VIDEO_PATH}")

# == Utils ==
def calculate_attention_entropy(attn_weights):
    # attn_weights shape: (batch, num_regions)
    epsilon = 1e-8
    attn = attn_weights + epsilon
    entropy = -(attn * attn.log()).sum(dim=1)  # per sample entropy
    return entropy.mean().item()  # average entropy over batch


def set_attention_trainable(model, trainable: bool):
    if hasattr(model, 'agva_attn'):
        for param in model.agva_attn.parameters():
            param.requires_grad = trainable
        logger.info(f"{'🔥 Unfroze' if trainable else '❄️ Froze'} attention parameters.")


def get_annealed_temperature(epoch, max_epoch, start_temp=1.0, end_temp=0.1):
    if epoch >= max_epoch:
        return end_temp
    return start_temp - (start_temp - end_temp) * (epoch / max_epoch)


def load_shallow_classifier_weights(attn_model, shallow_model_path):
    if not os.path.isfile(shallow_model_path):
        logger.warning(f"Shallow model weights file not found: {shallow_model_path}")
        return

    shallow_ckpt = torch.load(shallow_model_path, map_location=DEVICE)
    shallow_state = shallow_ckpt  # adapt if using checkpoint format

    attn_state = attn_model.state_dict()
    classifier_keys = [k for k in attn_state if k.startswith('classifier.')]
    filtered_shallow = {k: v for k, v in shallow_state.items() if k in classifier_keys}
    attn_state.update(filtered_shallow)
    attn_model.load_state_dict(attn_state)

    logger.info(f"Loaded shallow classifier weights from {shallow_model_path} into AG attention model.")

# == Dataset ==
logger.info("Loading datasets...")
train_ds = PerVideoMultimodalDataset(
    annotations_file=ANNOTATIONS_PATH,
    audio_h5_path=AUDIO_PATH,
    video_h5_path=VIDEO_PATH,
    split="train",
    modality=args.modality
)
val_ds = PerVideoMultimodalDataset(
    annotations_file=ANNOTATIONS_PATH,
    audio_h5_path=AUDIO_PATH,
    video_h5_path=VIDEO_PATH,
    split="val",
    modality=args.modality
)
test_ds = PerVideoMultimodalDataset(
    annotations_file=ANNOTATIONS_PATH,
    audio_h5_path=AUDIO_PATH,
    video_h5_path=VIDEO_PATH,
    split="test",
    modality=args.modality
)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=args.pin_memory)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.pin_memory)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=args.pin_memory)

# Instantiate model
if args.modality == "multimodal" and args.AGVattn:
    model = ModelClass(num_classes=NUM_CLASSES, style=args.style, debug=args.debug_attn, temperature=args.attn_temperature).to(DEVICE)

    # # Load shallow classifier weights (adapt path if needed)
    # shallow_path = os.path.join(
    #     "models/per_chunk", f"shallow_classifier_multimodal_features_{args.style}.pth"
    # )
    # load_shallow_classifier_weights(model, shallow_path)
    
else:
    model = ModelClass(num_classes=NUM_CLASSES, style=args.style).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(model)
logger.info(f"Total parameters: {num_params:,}")
logger.info(f"Trainable parameters: {num_trainable:,}")

# == Loss ==
if args.pos_weight:
    pos_weight = compute_pos_weight(train_ds, num_classes=NUM_CLASSES, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.BCEWithLogitsLoss()

# == Optimizer ==

if args.modality == "multimodal" and args.AGVattn:
    attn_params = list(model.agva_attn.parameters())
    classifier_params = [p for n, p in model.named_parameters() if not n.startswith('agva_attn.')]

    lr_classifier = args.lr 
    lr_attn = args.lr * 0.1
    optimizer = optim.AdamW([
        {'params': classifier_params, 'lr': lr_classifier},
        {'params': attn_params, 'lr': lr_attn}  # e.g. 10x lower LR for attention
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
    else:  # adam
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2)
        )
logger.info(f"Using optimizer: {optimizer}")

# == Scheduler ==
if args.use_scheduler:
    scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    logger.info(f"Using StepLR scheduler with step_size={args.scheduler_step_size}, gamma={args.scheduler_gamma}")
else:
    scheduler = None

# == Evaluation Function ==
@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    run_loss, run_h_acc, run_s_acc = 0.0, 0.0, 0.0

    # Initialize accumulators for attention stats
    if args.AGVattn and args.debug_attn:
        attn_entropy_sum = 0.0
        attn_max_sum = 0.0
        attn_min_sum = 0.0
        attn_batches = 0

    for batch in dataloader:
        if args.modality == "audio":
            X = batch["audio"].to(DEVICE)
            outputs = model(X)
        elif args.modality == "video":
            X = batch["video"].to(DEVICE).view(batch["video"].size(0), -1)
            outputs = model(X)
        else:
            X_audio = batch["audio"]
            X_video = batch["video"]

            if args.AGVattn and args.debug_attn:
                outputs, attn_weights, _ = model(X_audio.to(DEVICE), X_video.to(DEVICE))
                attn_entropy_sum += calculate_attention_entropy(attn_weights)
                attn_max_sum += attn_weights.max().item()
                attn_min_sum += attn_weights.min().item()
                attn_batches += 1
            else:
                outputs = model(X_audio.to(DEVICE), X_video.to(DEVICE))

        labels = batch["labels"].to(DEVICE)
        loss = criterion(outputs, labels)

        probas = torch.sigmoid(outputs)
        preds = (probas > 0.5).float()

        # Reshape
        preds_reshaped = preds.view(-1, NUM_CLASSES)
        labels_reshaped = labels.view(-1, NUM_CLASSES)

        # Transform to label lists
        pred_labels = [torch.nonzero(row).squeeze(1).tolist() for row in preds_reshaped]
        true_labels = [torch.nonzero(row).squeeze(1).tolist() for row in labels_reshaped]

        run_loss += loss.item()
        run_h_acc += hamming_accuracy_from_label_lists(true_labels, pred_labels)
        run_s_acc += subset_accuracy_from_label_lists(true_labels, pred_labels)

    if args.AGVattn and args.debug_attn and attn_batches > 0:
        avg_entropy = attn_entropy_sum / attn_batches
        avg_max = attn_max_sum / attn_batches
        avg_min = attn_min_sum / attn_batches
        logger.info(f"[DEBUG] Eval avg attention entropy: {avg_entropy:.4f}, max: {avg_max:.4f}, min: {avg_min:.4f}")

    n = len(dataloader)
    return run_loss / n, run_h_acc / n, run_s_acc / n

# == Training Function ==
def train(model, dataloader, criterion, optimizer):
    model.train()
    run_loss, run_h_acc, run_s_acc = 0.0, 0.0, 0.0

    # Initialize accumulators for attention stats
    if args.AGVattn and args.debug_attn: 
        attn_entropy_sum = 0.0
        attn_max_sum = 0.0
        attn_min_sum = 0.0
        attn_batches = 0

    for batch in tqdm(dataloader, desc="Training", ascii=True):
        if args.modality == "audio":
            X = batch["audio"].to(DEVICE)
            outputs = model(X)
        elif args.modality == "video":
            X = batch["video"].to(DEVICE)
            outputs = model(X)
        else:
            if args.AGVattn and args.debug_attn:
                outputs, attn_weights, attn_logits = model(batch["audio"].to(DEVICE), batch["video"].to(DEVICE))

                # Accumulate attention statistics
                attn_entropy_sum += calculate_attention_entropy(attn_weights)
                attn_max_sum += attn_weights.max().item()
                attn_min_sum += attn_weights.min().item()
                attn_batches += 1
        
            else:
                outputs = model(batch["audio"].to(DEVICE), batch["video"].to(DEVICE))

        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        probas = torch.sigmoid(outputs)
        preds = (probas > 0.5).float()

        # Reshape
        preds_reshaped = preds.view(-1, NUM_CLASSES)
        labels_reshaped = labels.view(-1, NUM_CLASSES)

        # Transform to label lists
        pred_labels = [torch.nonzero(row).squeeze(1).tolist() for row in preds_reshaped]
        true_labels = [torch.nonzero(row).squeeze(1).tolist() for row in labels_reshaped]

        run_loss += loss.item()
        run_h_acc += hamming_accuracy_from_label_lists(true_labels, pred_labels)
        run_s_acc += subset_accuracy_from_label_lists(true_labels, pred_labels)

    if args.AGVattn and args.debug_attn and attn_batches > 0:
        avg_entropy = attn_entropy_sum / attn_batches
        avg_max = attn_max_sum / attn_batches
        avg_min = attn_min_sum / attn_batches
        logger.info(f"[DEBUG] Avg attention entropy: {avg_entropy:.4f}, max: {avg_max:.4f}, min: {avg_min:.4f}")

    n = len(dataloader)
    return run_loss / n, run_h_acc / n, run_s_acc / n

# == Training Loop ==
def train_model():
    best_val_h_acc = 0.0
    patience_counter = 0
    hist = {
        'train_loss': [], 'val_loss': [],
        'train_h_acc': [], 'val_h_acc': [],
        'train_s_acc': [], 'val_s_acc': []
    }

    if args.modality == "multimodal" and args.AGVattn:
        max_anneal_epochs = 20
        anneal_start = WARMUP_EPOCHS
        anneal_end = WARMUP_EPOCHS + max_anneal_epochs
        set_attention_trainable(model, False)

    for epoch in range(args.epochs):
        logger.info(f"= Epoch {epoch+1}/{args.epochs} =")
        
        if args.modality == "multimodal" and args.AGVattn:
            if epoch == WARMUP_EPOCHS:
                set_attention_trainable(model, True)

            if args.no_anneal_temperature:
                # No annealing: keep temperature fixed at start value
                new_temp = args.attn_temperature
            else:
                # Existing annealing logic
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

        train_loss, train_h_acc, train_s_acc = train(model, train_loader, criterion, optimizer)
        val_loss, val_h_acc, val_s_acc = evaluate(model, val_loader, criterion)
        
        logger.info(f"[Train] Loss={train_loss:.4f} | HamAcc={train_h_acc:.4f} | SubAcc={train_s_acc:.4f}")
        logger.info(f"[Val]   Loss={val_loss:.4f} | HamAcc={val_h_acc:.4f} | SubAcc={val_s_acc:.4f}")
        
        hist['train_loss'].append(train_loss)
        hist['val_loss'].append(val_loss)
        hist['train_h_acc'].append(train_h_acc)
        hist['val_h_acc'].append(val_h_acc)
        hist['train_s_acc'].append(train_s_acc)
        hist['val_s_acc'].append(val_s_acc)

        if scheduler is not None:
            scheduler.step()
            logger.info(f"Learning rate after epoch {epoch+1}: {scheduler.get_last_lr()}")

        if val_h_acc > best_val_h_acc:
            best_val_h_acc = val_h_acc
            patience_counter = 0
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUTPUT_MODEL_PATH)
            logger.info(f"✅ Saved new best model with Val HamAcc={val_h_acc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"⚠️ No validation improvement @ PATIENCE ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                logger.info("🛑 Early stopping")
                break
    return hist

# == Train Plots ==
def save_training_plots(history, base_name, results_dir):
    """
    Save two plots:
      - losses: train_loss vs val_loss
      - accuracies: train_h_acc, val_h_acc, train_s_acc, val_s_acc
    """
    os.makedirs(results_dir, exist_ok=True)

    # --- Losses plot ---
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

    # --- Accuracies plot (Hamming + Subset) ---
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

# == Main ==
if __name__ == "__main__":

    # To train or not?
    if args.train:
        history = train_model()
        
        # Save training curves
        loss_path, acc_path = save_training_plots(
            history=history,
            base_name=BASE_NAME,
            results_dir=OUTPUT_FIGS_DIR
        )
        logger.info(f"Saved training loss plot to {loss_path}")
        logger.info(f"Saved training accuracy plot to {acc_path}")
    else:
        logger.info("Skipping training...")
    
    # Load pre-trained model
    if os.path.exists(OUTPUT_MODEL_PATH):
        model.load_state_dict(torch.load(OUTPUT_MODEL_PATH, map_location=DEVICE))
        logger.info(f"Loaded model from {OUTPUT_MODEL_PATH}")

    # Test model
    test_loss, test_h_acc, test_s_acc = evaluate(model, test_loader, criterion)
    logger.info(f"[Test] Loss={test_loss:.4f} | HamAcc={test_h_acc:.4f} | SubAcc={test_s_acc:.4f}")

    # Create a new row with the BASE_NAME and final metrics
    new_row = {
        "base_name": BASE_NAME,
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
        "pos_weight": args.pos_weight,
        "standardization": args.std
    }

    # Append to existing CSV or create a new one if it doesn't exist
    if os.path.exists(OUTPUT_CSV_PATH):
        df_results = pd.read_csv(OUTPUT_CSV_PATH)
        df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_results = pd.DataFrame([new_row])

    # Save back to CSV
    df_results.to_csv(OUTPUT_CSV_PATH, index=False)

    logger.info(f"Updated results table saved to {OUTPUT_CSV_PATH}")
    logger.info(f"\n{df_results.to_string(index=False)}")
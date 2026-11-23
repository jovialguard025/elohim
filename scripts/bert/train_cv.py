import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add current dir to path if needed
sys.path.insert(0, os.path.dirname(__file__))
from gap_data import load_gap, swap_ab, GapDataset
from model import SpanClassifier

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def evaluate(model, loader, device, use_amp, amp_dtype):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            a_mask = batch["a_mask"].to(device)
            b_mask = batch["b_mask"].to(device)
            p_mask = batch["p_mask"].to(device)
            labels = batch["label"].to(device)
            extra = batch["extra"].to(device) if "extra" in batch else None

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids, attention_mask, a_mask, b_mask, p_mask, extra=extra)
                loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    preds = np.argmax(probs_arr, axis=-1)
    acc = accuracy_score(targets_arr, preds)
    macro_f1 = f1_score(targets_arr, preds, average="macro")
    return avg_loss, acc, macro_f1, probs_arr

def main():
    parser = argparse.ArgumentParser(description="Train SpanClassifier on GAP using Stratified 5-Fold CV")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-large", help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--bs", type=int, default=4, help="Batch size per step")
    parser.add_argument("--accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr_enc", type=float, default=1e-5, help="Encoder learning rate")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Head learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--extra_features_dev", type=str, default=None, help="Path to dev extra features CSV")
    parser.add_argument("--extra_features_val", type=str, default=None, help="Path to val extra features CSV")
    args = parser.parse_args()

    seed_everything(args.seed)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results/bert", exist_ok=True)

    print(f"Loading GAP data for cross-validation...")
    dev_df = load_gap("data/raw/gap-development.tsv")
    val_df = load_gap("data/raw/gap-validation.tsv")
    full_df = pd.concat([dev_df, val_df], ignore_index=True)
    print(f"Combined Training Data: {len(full_df)} samples (dev={len(dev_df)}, val={len(val_df)})")

    # Extra features handling if provided
    extra_dev_mat = None
    extra_val_mat = None
    extra_full_mat = None
    n_extra = 0
    if args.extra_features_dev and args.extra_features_val:
        f_dev = pd.read_csv(args.extra_features_dev).drop(columns=["ID", "label"], errors="ignore").values
        f_val = pd.read_csv(args.extra_features_val).drop(columns=["ID", "label"], errors="ignore").values
        extra_full_mat = np.vstack([f_dev, f_val])
        n_extra = extra_full_mat.shape[1]
        print(f"Using {n_extra} extra engineered features.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    assert tokenizer.is_fast, f"Tokenizer {args.model} must be a FAST tokenizer!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using device: {device} | Mixed precision (AMP): {use_amp} (bf16={use_bf16})")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros((len(full_df), 3), dtype=np.float32)

    log_file = "results/bert/train_log.csv"
    with open(log_file, "w") as f:
        f.write("fold,epoch,train_loss,val_loss,val_acc,val_macro_f1\n")

    criterion = nn.CrossEntropyLoss()

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df.label)):
        print(f"\n{'='*30} FOLD {fold + 1} / {args.folds} {'='*30}")
        fold_train_df = full_df.iloc[train_idx].copy().reset_index(drop=True)
        fold_val_df = full_df.iloc[val_idx].copy().reset_index(drop=True)

        fold_train_extra = None
        fold_val_extra = None
        if extra_full_mat is not None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            tr_extra = extra_full_mat[train_idx]
            va_extra = extra_full_mat[val_idx]
            tr_extra_scaled = scaler.fit_transform(tr_extra)
            va_extra_scaled = scaler.transform(va_extra)
            # Duplicate for swap
            fold_train_extra = np.vstack([tr_extra_scaled, tr_extra_scaled])
            fold_val_extra = va_extra_scaled

        # Augmentation: swap A <-> B
        fold_train_swap = swap_ab(fold_train_df)
        fold_train_augmented = pd.concat([fold_train_df, fold_train_swap], ignore_index=True)
        print(f"Fold {fold}: Train size (with A<->B swap) = {len(fold_train_augmented)}, Val size = {len(fold_val_df)}")

        train_ds = GapDataset(fold_train_augmented, tokenizer, max_len=args.max_len, extra=fold_train_extra)
        val_ds = GapDataset(fold_val_df, tokenizer, max_len=args.max_len, extra=fold_val_extra)

        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.bs * 2, shuffle=False, num_workers=0)

        model = SpanClassifier(name=args.model, n_extra=n_extra).to(device)

        # Parameter groups
        enc_params = [p for p in model.enc.parameters() if p.requires_grad]
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": enc_params, "lr": args.lr_enc, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay}
        ])

        total_steps = (len(train_loader) // args.accum + (1 if len(train_loader) % args.accum != 0 else 0)) * args.epochs
        warmup_steps = int(0.10 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and not use_bf16))

        best_val_loss = float("inf")
        best_fold_probs = None
        best_model_path = f"models/bert_fold{fold}.pt"

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            running_loss = 0.0

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                a_mask = batch["a_mask"].to(device)
                b_mask = batch["b_mask"].to(device)
                p_mask = batch["p_mask"].to(device)
                labels = batch["label"].to(device)
                extra = batch["extra"].to(device) if "extra" in batch else None

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(input_ids, attention_mask, a_mask, b_mask, p_mask, extra=extra)
                    loss = criterion(logits, labels)
                    loss = loss / args.accum

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.item() * args.accum

                if (step + 1) % args.accum == 0 or (step + 1) == len(train_loader):
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            train_loss = running_loss / len(train_loader)
            val_loss, val_acc, val_f1, fold_val_probs = evaluate(model, val_loader, device, use_amp, amp_dtype)

            print(f"Fold {fold} Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")

            with open(log_file, "a") as f:
                f.write(f"{fold},{epoch},{train_loss:.6f},{val_loss:.6f},{val_acc:.6f},{val_f1:.6f}\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_fold_probs = fold_val_probs
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> Saved new best checkpoint to {best_model_path} (Val Loss: {val_loss:.4f})")

        oof_probs[val_idx] = best_fold_probs
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.save("results/bert/oof_probs.npy", oof_probs)
    oof_preds = np.argmax(oof_probs, axis=-1)
    targets = full_df.label.values
    oof_acc = accuracy_score(targets, oof_preds)
    oof_f1 = f1_score(targets, oof_preds, average="macro")

    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION SUMMARY (OOF)")
    print(f"Overall OOF Accuracy: {oof_acc * 100:.2f}%")
    print(f"Overall OOF Macro-F1: {oof_f1:.4f}")
    print(f"OOF Probabilities saved to results/bert/oof_probs.npy")
    print("=" * 60)

if __name__ == "__main__":
    main()

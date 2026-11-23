import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from gap_data import load_gap, swap_ab, GapDataset
from model import SpanClassifier

def predict_loader(model, loader, device, use_amp, amp_dtype):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            a_mask = batch["a_mask"].to(device)
            b_mask = batch["b_mask"].to(device)
            p_mask = batch["p_mask"].to(device)
            extra = batch["extra"].to(device) if "extra" in batch else None

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids, attention_mask, a_mask, b_mask, p_mask, extra=extra)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate 5-Fold Ensemble on GAP Test Set")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-large", help="HuggingFace model name")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to ensemble")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--bs", type=int, default=8, help="Batch size for test evaluation")
    parser.add_argument("--extra_features_train", type=str, default=None, help="Path to train extra features CSV")
    parser.add_argument("--extra_features_test", type=str, default=None, help="Path to test extra features CSV")
    args = parser.parse_args()

    os.makedirs("results/bert", exist_ok=True)

    print("Loading test dataset data/raw/gap-test.tsv...")
    test_df = load_gap("data/raw/gap-test.tsv")
    test_swap_df = swap_ab(test_df)
    print(f"Test samples: {len(test_df)}")

    test_extra = None
    n_extra = 0
    if args.extra_features_train and args.extra_features_test:
        from sklearn.preprocessing import StandardScaler
        tr_df = pd.read_csv(args.extra_features_train).drop(columns=["ID", "label"], errors="ignore").values
        te_df = pd.read_csv(args.extra_features_test).drop(columns=["ID", "label"], errors="ignore").values
        scaler = StandardScaler()
        scaler.fit(tr_df)
        test_extra = scaler.transform(te_df)
        n_extra = test_extra.shape[1]
        print(f"Loaded {n_extra} extra engineered features for test set.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    assert tokenizer.is_fast, f"Tokenizer {args.model} must be a FAST tokenizer!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using device: {device} | Mixed precision: {use_amp} (bf16={use_bf16})")

    test_ds = GapDataset(test_df, tokenizer, max_len=args.max_len, extra=test_extra)
    test_swap_ds = GapDataset(test_swap_df, tokenizer, max_len=args.max_len, extra=test_extra)

    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=0)
    test_swap_loader = DataLoader(test_swap_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    accumulated_probs = np.zeros((len(test_df), 3), dtype=np.float64)

    for fold in range(args.folds):
        ckpt_path = f"models/bert_fold{fold}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found. Please train models first.")

        print(f"Evaluating fold {fold + 1}/{args.folds} checkpoint {ckpt_path}...")
        model = SpanClassifier(name=args.model, n_extra=n_extra)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)

        # Standard forward pass
        p = predict_loader(model, test_loader, device, use_amp, amp_dtype)

        # Symmetrical test-time augmentation (swap A <-> B)
        p_swap_raw = predict_loader(model, test_swap_loader, device, use_amp, amp_dtype)
        # Swap columns back: col 0 becomes col 1, col 1 becomes col 0, col 2 stays 2
        p_swap = p_swap_raw[:, [1, 0, 2]]

        p_k = (p + p_swap) / 2.0
        accumulated_probs += p_k

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_probs = accumulated_probs / float(args.folds)
    np.save("results/bert/test_probs.npy", final_probs)

    preds = np.argmax(final_probs, axis=-1)
    targets = test_df.label.values

    accuracy = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro"))
    prec, rec, f1, sup = precision_recall_fscore_support(targets, preds, labels=[0, 1, 2])
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2])

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET EVALUATION REPORT")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print("\nPer-Class Breakdown:")
    class_names = ["Candidate A (0)", "Candidate B (1)", "Neither (2)"]
    per_class_dict = {}
    for c in range(3):
        print(f"  {class_names[c]:<18}: Precision = {prec[c]:.4f}, Recall = {rec[c]:.4f}, F1 = {f1[c]:.4f}, Support = {sup[c]}")
        per_class_dict[class_names[c]] = {
            "precision": float(prec[c]),
            "recall": float(rec[c]),
            "f1": float(f1[c]),
            "support": int(sup[c])
        }

    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    print("                 Predicted A  Predicted B  Predicted Neither")
    for i, row_name in enumerate(["Actual A      ", "Actual B      ", "Actual Neither"]):
        print(f"{row_name}   {cm[i][0]:>11}  {cm[i][1]:>11}  {cm[i][2]:>17}")

    result_json = {
        "model": args.model,
        "folds": args.folds,
        "test_samples": len(test_df),
        "test_accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_metrics": per_class_dict,
        "confusion_matrix": cm.tolist()
    }

    with open("results/bert/test_result.json", "w") as f:
        json.dump(result_json, f, indent=2)

    print(f"\nSaved test results to results/bert/test_result.json and results/bert/test_probs.npy")

if __name__ == "__main__":
    main()

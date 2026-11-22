import os
import csv
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_all():
    os.makedirs("results/comparisons", exist_ok=True)
    os.makedirs("results/experiments", exist_ok=True)
    
    # 1. Plot Loss curves for Best Model (Exp 4 & Exp 8)
    exp4_csv = "results/experiments/exp4_history.csv"
    if os.path.exists(exp4_csv):
        epochs, tr_loss, val_loss, val_acc, val_f1 = [], [], [], [], []
        with open(exp4_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                tr_loss.append(float(row['train_loss']))
                val_loss.append(float(row['val_loss']))
                val_acc.append(float(row['val_acc']))
                val_f1.append(float(row['val_macro_f1']))

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, tr_loss, label='Training Loss (Weighted CE)', color='#1f77b4', lw=2)
        plt.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', lw=2)
        plt.title('Training and Validation Loss: Experiment 4 (Best Square-Root Weighting)', fontsize=13)
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel('Weighted Cross-Entropy Loss', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig("results/comparisons/training_validation_loss.png", dpi=300)
        plt.close()
        print("Generated results/comparisons/training_validation_loss.png")

        # 2. Plot Accuracy and Macro-F1 progression
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, [a * 100 for a in val_acc], label='Validation Accuracy (%)', color='#2ca02c', lw=2)
        plt.plot(epochs, [f * 100 for f in val_f1], label='Validation Macro-F1 (%)', color='#d62728', lw=2)
        plt.title('Validation Accuracy & Macro-F1 vs. Epochs: Experiment 4', fontsize=13)
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel('Percentage (%)', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig("results/comparisons/validation_metrics_progression.png", dpi=300)
        plt.close()
        print("Generated results/comparisons/validation_metrics_progression.png")

    # 3. Plot Confusion Matrix Heatmap for Best Model
    exp4_json = "results/experiments/exp4_result.json"
    if os.path.exists(exp4_json):
        with open(exp4_json, 'r') as f:
            data = json.load(f)
        cm = np.array(data['confusion_matrix'])
        classes = ['A', 'B', 'Neither']

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap='Blues')
        fig.colorbar(cax)

        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > np.max(cm)/2 else 'black',
                        fontsize=12, fontweight='bold')

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(classes, fontsize=11)
        ax.set_yticklabels(classes, fontsize=11)
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('Actual Class', fontsize=12)
        ax.set_title('Confusion Matrix: Best Model (Val Acc: 63.44%)', pad=15, fontsize=13)
        plt.tight_layout()
        plt.savefig("results/comparisons/confusion_matrix_best_model.png", dpi=300)
        plt.close()
        print("Generated results/comparisons/confusion_matrix_best_model.png")

    # 4. Plot Comparison Bar Chart: Reproduced vs. Reference
    cmp_csv = "results/comparisons/experiment_comparison.csv"
    if os.path.exists(cmp_csv):
        exp_names, reprod_f1s, ref_f1s = [], [], []
        short_names = [
            "Exp 1\n(12-feat)", "Exp 2\n(Alpha=1.0)", "Exp 3\n(LR=3e-4)",
            "Exp 4\n(Alpha=0.5)", "Exp 5\n(Alpha=0.65)", "Exp 6\n(30-feat)",
            "Exp 7\n(Hid=64)", "Exp 8\n(L2=1e-4)"
        ]
        with open(cmp_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp_names.append(row['Experiment'])
                reprod_f1s.append(float(row['Reprod_MacroF1']))
                ref_f1s.append(float(row['Ref_MacroF1']))

        x = np.arange(len(short_names))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, ref_f1s, width, label='Historical Reference Macro-F1', color='#aec7e8')
        plt.bar(x + width/2, reprod_f1s, width, label='Reproduced Baseline Macro-F1', color='#1f77b4')

        plt.ylabel('Macro-F1 Score', fontsize=12)
        plt.title('Macro-F1 Across All 8 Experiments: Historical Reference vs. Reproduced Baseline', fontsize=13)
        plt.xticks(x, short_names, fontsize=10)
        plt.ylim(0.40, 0.62)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig("results/comparisons/experiment_f1_comparison.png", dpi=300)
        plt.close()
        print("Generated results/comparisons/experiment_f1_comparison.png")

if __name__ == '__main__':
    plot_all()

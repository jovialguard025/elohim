import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

def run_gbm_diagnostic():
    print("Running Phase 5 Ceiling Diagnostic: Gradient Boosting (scikit-learn)")
    train_feat_path = "data/processed/dev_val_best_feat.csv"
    if not os.path.exists(train_feat_path):
        train_feat_path = "data/processed/dev_val_36feat.csv"
    if not os.path.exists(train_feat_path):
        print(f"Error: {train_feat_path} does not exist.")
        return

    df = pd.read_csv(train_feat_path)
    X = df.drop(columns=["ID", "label"]).values
    y = df["label"].values
    n = len(y)

    seeds = [42, 43, 44]
    acc_scores = []
    f1_scores = []
    loss_scores = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train)
            X_va_sc = scaler.transform(X_val)

            clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, random_state=seed)
            clf.fit(X_tr_sc, y_train)

            probs = clf.predict_proba(X_va_sc)
            preds = np.argmax(probs, axis=1)

            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds, average="macro")
            loss = log_loss(y_val, probs)

            acc_scores.append(acc)
            f1_scores.append(f1)
            loss_scores.append(loss)

    acc_mean = np.mean(acc_scores)
    acc_std = np.std(acc_scores)
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    loss_mean = np.mean(loss_scores)
    loss_std = np.std(loss_scores)

    print("\n--- Gradient Boosting CV Results (15 folds) ---")
    print(f"GBM CV Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
    print(f"GBM CV Macro-F1: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"GBM CV Log-Loss: {loss_mean:.4f} ± {loss_std:.4f}")

    os.makedirs("results/experiments", exist_ok=True)
    with open("results/experiments/gbm_ceiling_diagnostic.txt", "w") as f:
        f.write("Gradient Boosting Classifier Ceiling Diagnostic (scikit-learn)\n")
        f.write(f"15-fold CV Accuracy: {acc_mean*100:.2f}% +/- {acc_std*100:.2f}%\n")
        f.write(f"15-fold CV Macro-F1: {f1_mean:.4f} +/- {f1_std:.4f}\n")
        f.write(f"15-fold CV Log-Loss: {loss_mean:.4f} +/- {loss_std:.4f}\n")

if __name__ == "__main__":
    run_gbm_diagnostic()

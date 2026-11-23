import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def load_gap(path):
    df = pd.read_csv(path, sep="\t")
    df["label"] = 2
    a_coref = (df["A-coref"] == True) | (df["A-coref"].astype(str).str.upper() == "TRUE")
    b_coref = (df["B-coref"] == True) | (df["B-coref"].astype(str).str.upper() == "TRUE")
    df.loc[a_coref, "label"] = 0
    df.loc[b_coref, "label"] = 1
    return df.reset_index(drop=True)

def swap_ab(df):
    """Augmentation: swap candidates A and B, labels 0<->1. Doubles training data."""
    df = df.reset_index(drop=True)
    s = df.copy()
    for c in ["", "-offset", "-coref"]:
        s["A" + c], s["B" + c] = df["B" + c].values, df["A" + c].values
    s["label"] = df["label"].map({0: 1, 1: 0, 2: 2})
    return s

class GapDataset(Dataset):
    def __init__(self, df, tok, max_len=512, extra=None):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.max_len = max_len
        self.extra = extra

    def __len__(self):
        return len(self.df)

    def _mask(self, offsets, start, length):
        end = start + length
        m = torch.zeros(self.max_len, dtype=torch.float32)
        for k, (s, e) in enumerate(offsets):
            if e > s and e > start and s < end:
                m[k] = 1.0
        return m

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(
            r.Text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length"
        )
        off = enc["offset_mapping"]
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "a_mask": self._mask(off, int(r["A-offset"]), len(r.A)),
            "b_mask": self._mask(off, int(r["B-offset"]), len(r.B)),
            "p_mask": self._mask(off, int(r["Pronoun-offset"]), len(r.Pronoun)),
            "label": torch.tensor(int(r.label), dtype=torch.long),
        }
        if self.extra is not None:
            item["extra"] = torch.tensor(self.extra[i], dtype=torch.float32)
        return item

def sanity_check(model_name="microsoft/deberta-v3-large"):
    print(f"Running sanity check with tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    assert tok.is_fast, f"Tokenizer {model_name} must be a FAST tokenizer (offset_mapping required)!"

    dev_path = "data/raw/gap-development.tsv"
    val_path = "data/raw/gap-validation.tsv"
    test_path = "data/raw/gap-test.tsv"

    dev = load_gap(dev_path)
    val = load_gap(val_path)
    test = load_gap(test_path)
    all_df = pd.concat([dev, val, test], ignore_index=True)
    total_rows = len(all_df)
    print(f"Loaded {total_rows} total rows across dev, val, test.")
    assert total_rows == 4454, f"Expected 4454 rows, got {total_rows}"

    dataset = GapDataset(all_df, tok, max_len=512)
    fail_a, fail_b, fail_p = 0, 0, 0
    for idx in range(total_rows):
        item = dataset[idx]
        if item["a_mask"].sum().item() <= 0:
            fail_a += 1
        if item["b_mask"].sum().item() <= 0:
            fail_b += 1
        if item["p_mask"].sum().item() <= 0:
            fail_p += 1

    print(f"Sanity Check Results: Fail A: {fail_a}, Fail B: {fail_b}, Fail P: {fail_p}")
    assert fail_a == 0, f"{fail_a} rows had empty A mask!"
    assert fail_b == 0, f"{fail_b} rows had empty B mask!"
    assert fail_p == 0, f"{fail_p} rows had empty Pronoun mask!"
    print("ALL 4,454 SPAN MASKS VALIDATED SUCCESSFULLY.")

if __name__ == "__main__":
    sanity_check()

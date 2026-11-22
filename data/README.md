# Data Directory (`data/`)

> **Policy: Upload Selectively**  
> **Keep raw data out initially unless its inclusion is explicitly justified.**  
> Large raw dumps, binary archives, and heavy unformatted datasets must not be committed directly into Git. Use download scripts or external cloud storage pointers instead.

---

## 1. Dataset Source Information & Attribution
- **Dataset Name**: *(Document benchmark datasets here, e.g., GSM8K, HotpotQA, Multi-Step Reasoning benchmarks)*
- **Source URL**: *(Link to primary source repository or HuggingFace hub)*
- **License / Terms of Use**: *(Document original source licensing and attribution requirements)*

---

## 2. Download Instructions
To fetch the raw benchmark data locally:
```bash
# Example download routine via script
python scripts/prepare_data.py --input <source_url_or_file> --output data/raw/
```

---

## 3. Expected Split Sizes
| Split | Target Percentage | Expected Record Count | Description |
| :--- | :--- | :--- | :--- |
| **Train** | 80% | TBD | Calibration of confidence gating thresholds |
| **Validation** | 10% | TBD | Hyperparameter tuning and threshold cutoff selection |
| **Test** | 10% | TBD | Final evaluation and context-budget benchmark |

---

## 4. Approved Derived Datasets
Approved derived datasets stored under `data/processed/` must be:
- Sanitized (no private API tokens or identifiable private information).
- Compact (JSON / CSV formatted, suitable for version control or LFS).
- Accompanied by a record of preprocessing parameters.

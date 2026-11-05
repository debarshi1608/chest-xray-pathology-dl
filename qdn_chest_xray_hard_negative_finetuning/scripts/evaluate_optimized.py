# Source: Legendary_DL_report.pdf, section 8.1
# Stage-1 abnormality gate + validation-tuned thresholds; evaluates F1 / AUPRC / AUROC on the held-out test split.
import os
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

FOCUS_CLASSES = ["Effusion", "Pneumothorax", "Cardiomegaly"]
GATE_THRESHOLD = 0.30

def evaluate_model_dir(base_dir="outputs/final/qdn", gate_thresh=GATE_THRESHOLD):
    print(f"=== Evaluating {base_dir} (Stage-1 Gate Threshold = {gate_thresh}) ===")
    val_gate = np.load(os.path.join(base_dir, "gate", "val_probs.npy"))
    test_gate = np.load(os.path.join(base_dir, "gate", "test_probs.npy"))

    for cls_name in FOCUS_CLASSES:
        cls_dir = os.path.join(base_dir, f"cls_{cls_name}")
        val_p = np.load(os.path.join(cls_dir, "val_probs.npy"))
        val_y = np.load(os.path.join(cls_dir, "val_labels.npy"))
        test_p = np.load(os.path.join(cls_dir, "test_probs.npy"))
        test_y = np.load(os.path.join(cls_dir, "test_labels.npy"))

        # Apply Stage-1 Gate override on validation predictions
        val_p_gated = np.where(val_gate < gate_thresh, 0.0, val_p)
        best_th, best_val_f1 = 0.5, 0.0
        for th in np.linspace(0.05, 0.95, 91):
            f1 = f1_score(val_y, (val_p_gated >= th).astype(int), zero_division=0)
            if f1 > best_val_f1:
                best_val_f1, best_th = f1, th

        # Evaluate on held-out test split using validation-calibrated threshold
        test_p_gated = np.where(test_gate < gate_thresh, 0.0, test_p)
        test_preds = (test_p_gated >= best_th).astype(int)
        test_f1 = f1_score(test_y, test_preds, zero_division=0)
        test_auprc = average_precision_score(test_y, test_p)
        test_auroc = roc_auc_score(test_y, test_p)
        tp = int(((test_preds == 1) & (test_y == 1)).sum())
        fp = int(((test_preds == 1) & (test_y == 0)).sum())

        print(f"{cls_name:15s} | Test F1: {test_f1:.4f} (Val F1: {best_val_f1:.4f} @ th={best_th:.2f}) "
              f"| TP: {tp:4d} | FP: {fp:4d} | AUPRC: {test_auprc:.4f} | AUROC: {test_auroc:.4f}")

if __name__ == "__main__":
    evaluate_model_dir("outputs/final/qdn")

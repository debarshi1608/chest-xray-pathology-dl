
import os
import numpy as np
import pandas as pd

FOCUS_CLASSES = ["Effusion", "Pneumothorax", "Cardiomegaly"]

def inspect_false_positives(base_dir="outputs/final/qdn", test_csv="splits/test.csv", top_k=10):
    df_test = pd.read_csv(test_csv)
    test_gate = np.load(os.path.join(base_dir, "gate", "test_probs.npy"))

    for cls_name in FOCUS_CLASSES:
        cls_dir = os.path.join(base_dir, f"cls_{cls_name}")
        test_p = np.load(os.path.join(cls_dir, "test_probs.npy"))
        test_y = np.load(os.path.join(cls_dir, "test_labels.npy"))

        # Isolate negative ground-truth instances sorted by predicted probability
        neg_indices = np.where(test_y == 0)[0]
        sorted_fp_idx = neg_indices[np.argsort(test_p[neg_indices])[::-1]][:top_k]

        print(f"\n=== Top {top_k} False Positives for {cls_name} ===")
        for idx in sorted_fp_idx:
            row = df_test.iloc[idx]
            img_id = row.get("Image Index", idx)
            view = row.get("View Position", "N/A")
            abn = int(row.get("Is_Abnormal", 0))
            findings = row.get("Finding Labels", "N/A")
            print(f"Image: {img_id} | Prob: {test_p[idx]:.4f} | Gate: {test_gate[idx]:.4f} "
                  f"| View: {view} | Is_Abnormal: {abn} | True Findings: {findings}")

if __name__ == "__main__":
    inspect_false_positives()

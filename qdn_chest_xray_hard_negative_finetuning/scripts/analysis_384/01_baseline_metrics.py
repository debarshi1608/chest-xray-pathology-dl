# Source: output_boxes PDF, section 3.1 "Script 1: Baseline Raw Metrics Across All Modules"
# Reconstructed from the chat export (output_boxes PDF). Lines cut off at the page edge were completed from the other copies of the same snippet
# and from the printed tables; those lines are marked "# [restored]". Check against your shell history (history | grep python3) before relying on it.
# Run from the repo root: python3 01_baseline_metrics.py
import os, glob, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

base = 'outputs/final/qdn_384'
dirs = ['gate'] + sorted([os.path.basename(d) for d in glob.glob(f'{base}/cls_*')])
print(f'{"Module":<22} | {"Val AUPRC":>9} | {"Test AUROC":>10} | {"Test AUPRC":>10} | {"Val-Th":>6} | {"Test F1":>7} | {"Test Prev":>9}')  # [restored]
print('-'*92)
for d in dirs:
    p = os.path.join(base, d)
    vy, vp = np.load(f'{p}/val_labels.npy'), np.load(f'{p}/val_probs.npy')
    ty, tp = np.load(f'{p}/test_labels.npy'), np.load(f'{p}/test_probs.npy')
    val_ap = average_precision_score(vy, vp)
    test_auc = roc_auc_score(ty, tp)
    test_ap = average_precision_score(ty, tp)
    prec, rec, th = precision_recall_curve(vy, vp)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_th = th[np.argmax(f1s[:-1])] if len(th) > 0 else 0.5
    preds = (tp >= best_th).astype(int)
    tp_c = np.sum((preds == 1) & (ty == 1))
    fp_c = np.sum((preds == 1) & (ty == 0))
    fn_c = np.sum((preds == 0) & (ty == 1))
    test_f1 = 2 * tp_c / (2 * tp_c + fp_c + fn_c + 1e-8)
    prev = np.mean(ty) * 100
    print(f'{d:<22} | {val_ap:9.4f} | {test_auc:10.4f} | {test_ap:10.4f} | {best_th:6.2f} | {test_f1:7.4f} | {prev:9.2f}%')

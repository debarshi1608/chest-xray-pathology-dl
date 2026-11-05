# Source: output_boxes PDF, section 7.2 "Script 5" (the corrected here-document version, run with: python3 - << 'EOF' ... EOF)
# Reconstructed from the chat export (output_boxes PDF). Lines cut off at the page edge were completed from the other copies of the same snippet
# and from the printed tables; those lines are marked "# [restored]". Check against your shell history (history | grep python3) before relying on it.
# Natural-prevalence F1, Youden sensitivity/specificity, and validation-locked 1:1 case-control F1
# (vs all negatives, and vs healthy scans only). The 1:1 F1 is NOT comparable to the natural-prevalence F1: report both and label them.
import os, glob, csv, numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

base = 'outputs/final/qdn_384'
np.random.seed(42)
N_BOOT = 50
dirs = sorted([os.path.basename(d) for d in glob.glob(f'{base}/cls_*')])
gv_abn = np.load(f'{base}/gate/val_labels.npy')
gv_prob = np.load(f'{base}/gate/val_probs.npy')
gt_abn = np.load(f'{base}/gate/test_labels.npy')
gt_prob = np.load(f'{base}/gate/test_probs.npy')

def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    return th[np.argmax((2*pr*rc/(pr+rc+1e-8))[:-1])]

def f1(pred, y):
    return 2*np.sum((pred==1)&(y==1)) / (np.sum(pred)+np.sum(y)+1e-8)

rows = []
for d in dirs:
    cname = d.replace('cls_', '')
    vy = np.load(f'{base}/{d}/val_labels.npy')
    vp = np.load(f'{base}/{d}/val_probs.npy') * (gv_prob ** 0.3)
    ty = np.load(f'{base}/{d}/test_labels.npy')
    tp = np.load(f'{base}/{d}/test_probs.npy') * (gt_prob ** 0.3)
    auc = roc_auc_score(ty, tp)

    # 1. Natural F1 (validation-locked) and Youden operating point
    nat_th = best_f1_threshold(vy, vp)
    nat_pred = (tp >= nat_th).astype(int)
    nat_f1 = f1(nat_pred, ty)
    fpr, tpr, r_th = roc_curve(vy, vp)
    j_th = r_th[np.argmax(tpr - fpr)]
    j_pred = (tp >= j_th).astype(int)
    sens = np.sum((j_pred==1)&(ty==1)) / np.sum(ty==1)
    spec = np.sum((j_pred==0)&(ty==0)) / np.sum(ty==0)
    acc = np.mean(nat_pred == ty)

    # 2. 1:1 case-control vs all negatives (threshold locked on a 1:1 validation subsample)
    v_pos, v_neg = np.where(vy == 1)[0], np.where(vy == 0)[0]
    v_sub_neg = np.random.choice(v_neg, size=len(v_pos), replace=False)
    v_bal_idx = np.concatenate([v_pos, v_sub_neg])
    bal_th = best_f1_threshold(vy[v_bal_idx], vp[v_bal_idx])
    t_pos, t_neg = np.where(ty == 1)[0], np.where(ty == 0)[0]
    bf1s = []
    for _ in range(N_BOOT):
        t_sub_neg = np.random.choice(t_neg, size=len(t_pos), replace=False)
        idx = np.concatenate([t_pos, t_sub_neg])
        bf1s.append(f1((tp[idx] >= bal_th).astype(int), ty[idx]))
    bal_f1 = float(np.mean(bf1s))

    # 3. 1:1 case-control vs healthy normal lungs only
    vn_neg = np.where((vy == 0) & (gv_abn == 0))[0]
    vn_sub = np.random.choice(vn_neg, size=min(len(v_pos), len(vn_neg)), replace=False)
    vn_idx = np.concatenate([v_pos, vn_sub])
    norm_th = best_f1_threshold(vy[vn_idx], vp[vn_idx])
    tn_neg = np.where((ty == 0) & (gt_abn == 0))[0]
    norm_auc = roc_auc_score(np.concatenate([ty[t_pos], ty[tn_neg]]), np.concatenate([tp[t_pos], tp[tn_neg]]))  # [restored]
    nf1s = []
    for _ in range(N_BOOT):
        tn_sub = np.random.choice(tn_neg, size=min(len(t_pos), len(tn_neg)), replace=False)
        idx = np.concatenate([t_pos, tn_sub])
        nf1s.append(f1((tp[idx] >= norm_th).astype(int), ty[idx]))
    norm_f1 = float(np.mean(nf1s))

    rows.append({
        'Class': cname, 'AUROC': round(auc, 4), 'Sensitivity': round(float(sens), 4), 'Specificity': round(float(spec), 4),
        'Accuracy': round(float(acc), 4), 'Nat_F1': round(float(nat_f1), 4), 'Bal_1to1_F1': round(bal_f1, 4),
        'Vs_Normal_AUROC': round(norm_auc, 4), 'Vs_Normal_1to1_F1': round(norm_f1, 4)
    })

with open(f'{base}/benchmark_comparison_table.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
for k in ['AUROC', 'Sensitivity', 'Accuracy', 'Nat_F1', 'Bal_1to1_F1', 'Vs_Normal_AUROC', 'Vs_Normal_1to1_F1']:
    print(k, round(float(np.mean([r[k] for r in rows])), 4))

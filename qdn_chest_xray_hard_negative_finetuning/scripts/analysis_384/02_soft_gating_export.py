
# Writes outputs/final/qdn_384/full_11class_metrics.json and .csv
import os, glob, json, csv, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

base = 'outputs/final/qdn_384'

# 1. Verify hard vs soft gate on Effusion
gp = np.load(f'{base}/gate/test_probs.npy')
ep = np.load(f'{base}/cls_Effusion/test_probs.npy')
ey = np.load(f'{base}/cls_Effusion/test_labels.npy')
print('Min Effusion prob when gate < 0.3:', ep[gp < 0.3].min(), 'Max:', ep[gp < 0.3].max())  # [restored]
print('Raw Effusion AUPRC:               ', round(average_precision_score(ey, ep), 4))
print('Hard-gated (gate>=0.3) AUPRC:     ', round(average_precision_score(ey, ep * (gp >= 0.3)), 4))  # [restored]
print('Soft-gated (gate * cls) AUPRC:    ', round(average_precision_score(ey, ep * gp), 4))  # [restored]

# 2. Tune the soft-gate exponent alpha on validation and export the full 11-class table
gv, gt = np.load(f'{base}/gate/val_probs.npy'), np.load(f'{base}/gate/test_probs.npy')
dirs = sorted([os.path.basename(d) for d in glob.glob(f'{base}/cls_*')])
results, rows = {}, []
for d in dirs:
    cname = d.replace('cls_', '')
    vy, vp = np.load(f'{base}/{d}/val_labels.npy'), np.load(f'{base}/{d}/val_probs.npy')
    ty, tp = np.load(f'{base}/{d}/test_labels.npy'), np.load(f'{base}/{d}/test_probs.npy')
    raw_auc = float(roc_auc_score(ty, tp))
    raw_ap = float(average_precision_score(ty, tp))
    p, r, th = precision_recall_curve(vy, vp)
    f1 = 2*p*r/(p+r+1e-8)
    r_th = float(th[np.argmax(f1[:-1])])
    r_pred = (tp >= r_th).astype(int)
    r_f1 = float(2*np.sum((r_pred==1)&(ty==1)) / (np.sum(r_pred)+np.sum(ty)+1e-8))

    best_a, best_val_ap = 0.0, -1.0
    for a in np.linspace(0, 1.5, 16):
        vap = average_precision_score(vy, vp * (gv ** a))
        if vap > best_val_ap:
            best_val_ap, best_a = float(vap), float(round(a, 2))
    svp, stp = vp * (gv ** best_a), tp * (gt ** best_a)
    soft_auc = float(roc_auc_score(ty, stp))
    soft_ap = float(average_precision_score(ty, stp))
    p_s, r_s, th_s = precision_recall_curve(vy, svp)
    f1_s = 2*p_s*r_s/(p_s+r_s+1e-8)
    s_th = float(th_s[np.argmax(f1_s[:-1])])
    s_pred = (stp >= s_th).astype(int)
    s_f1 = float(2*np.sum((s_pred==1)&(ty==1)) / (np.sum(s_pred)+np.sum(ty)+1e-8))
    tp_c = int(np.sum((s_pred==1)&(ty==1)))
    fp_c = int(np.sum((s_pred==1)&(ty==0)))
    fn_c = int(np.sum((s_pred==0)&(ty==1)))
    entry = {  # [restored: key order follows the printed table header]
        'Class': cname, 'Prevalence_%': round(float(np.mean(ty)*100), 2),
        'Raw_AUROC': round(raw_auc, 4), 'Raw_AUPRC': round(raw_ap, 4), 'Raw_F1': round(r_f1, 4), 'Raw_Val_Th': round(r_th, 4),
        'Soft_Alpha': best_a, 'Soft_AUROC': round(soft_auc, 4), 'Soft_AUPRC': round(soft_ap, 4),
        'Soft_F1': round(s_f1, 4), 'Soft_Val_Th': round(s_th, 4),
        'Soft_TP': tp_c, 'Soft_FP': fp_c, 'Soft_FN': fn_c
    }
    results[cname] = entry
    rows.append(entry)

with open(f'{base}/full_11class_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
with open(f'{base}/full_11class_metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print('Saved outputs/final/qdn_384/full_11class_metrics.json and .csv successfully.')

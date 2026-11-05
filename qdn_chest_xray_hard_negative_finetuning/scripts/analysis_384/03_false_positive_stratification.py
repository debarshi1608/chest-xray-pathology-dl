# Source: output_boxes PDF, section 5.1 "Script 3: False Positive Stratification (Normal Lungs vs. Other Pathologies)"
# Reconstructed from the chat export (output_boxes PDF). Lines cut off at the page edge were completed from the other copies of the same snippet
# and from the printed tables; those lines are marked "# [restored]". Check against your shell history (history | grep python3) before relying on it.
# Needs full_11class_metrics.json from 02_soft_gating_export.py
import json, numpy as np

base = 'outputs/final/qdn_384'
gt_abn = np.load(f'{base}/gate/test_labels.npy')
gt_prob = np.load(f'{base}/gate/test_probs.npy')
with open(f'{base}/full_11class_metrics.json') as f: meta = json.load(f)
print(f'{"Class":<20} | {"Soft_FP":>7} | {"FP on Normal Lungs":>18} | {"FP on Other Pathologies":>23}')  # [restored]
print('-'*76)
for c, m in meta.items():
    ty = np.load(f'{base}/cls_{c}/test_labels.npy')
    tp = np.load(f'{base}/cls_{c}/test_probs.npy') * (gt_prob ** m['Soft_Alpha'])
    fp_mask = (tp >= m['Soft_Val_Th']) & (ty == 0)
    tot, norm, oth = int(fp_mask.sum()), int((fp_mask & (gt_abn==0)).sum()), int((fp_mask & (gt_abn==1)).sum())  # [restored]
    print(f'{c:<20} | {tot:7d} | {norm:10d} ({norm/tot*100:4.1f}%) | {oth:15d} ({oth/tot*100:4.1f}%)')  # [restored]

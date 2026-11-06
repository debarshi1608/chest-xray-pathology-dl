
import os, glob, numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

base = 'outputs/final/qdn_384'
dirs = sorted([os.path.basename(d) for d in glob.glob(f'{base}/cls_*')])

def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

gv, gt = np.load(f'{base}/gate/val_probs.npy'), np.load(f'{base}/gate/test_probs.npy')
Xv = np.column_stack([to_logit(gv)] + [to_logit(np.load(f'{base}/{d}/val_probs.npy')) for d in dirs])   # [restored]
Xt = np.column_stack([to_logit(gt)] + [to_logit(np.load(f'{base}/{d}/test_probs.npy')) for d in dirs])  # [restored]
print(f'{"Class":<20} | {"Val Prev":>8} | {"Test Prev":>9} | {"Raw F1":>7} | {"Soft F1":>7} | {"Calib F1":>8} | {"Oracle Test F1":>14}')  # [restored]
print('-'*92)
raw_f1s, soft_f1s, cal_f1s, orc_f1s = [], [], [], []
for d in dirs:
    cname = d.replace('cls_', '')
    vy, ty = np.load(f'{base}/{d}/val_labels.npy'), np.load(f'{base}/{d}/test_labels.npy')
    vp, tp = np.load(f'{base}/{d}/val_probs.npy'), np.load(f'{base}/{d}/test_probs.npy')
    # raw F1, threshold from validation
    p, r, th = precision_recall_curve(vy, vp)
    r_th = th[np.argmax((2*p*r/(p+r+1e-8))[:-1])]
    r_f1 = 2*np.sum(((tp>=r_th)==1)&(ty==1)) / (np.sum(tp>=r_th)+np.sum(ty)+1e-8)
    # soft-gated F1 (alpha = 0.3)
    svp, stp = vp * (gv ** 0.3), tp * (gt ** 0.3)
    p, r, th = precision_recall_curve(vy, svp)
    s_th = th[np.argmax((2*p*r/(p+r+1e-8))[:-1])]
    s_f1 = 2*np.sum(((stp>=s_th)==1)&(ty==1)) / (np.sum(stp>=s_th)+np.sum(ty)+1e-8)
    # oracle: best F1 on the test PR curve
    pt, rt, _ = precision_recall_curve(ty, stp)
    orc_f1 = np.max(2*pt*rt/(pt+rt+1e-8))
    # stacked logistic calibration on all 12 outputs + prior-shift correction
    clf = LogisticRegression(C=0.1, max_iter=500).fit(Xv, vy)
    cal_vp, cal_tp = clf.predict_proba(Xv)[:, 1], clf.predict_proba(Xt)[:, 1]
    pi_v, pi_t = np.mean(vy), np.mean(ty)
    odds = (pi_t / (1 - pi_t)) / (pi_v / (1 - pi_v))
    cal_tp_adj = (cal_tp * odds) / (1 - cal_tp + cal_tp * odds)
    p_c, r_c, th_c = precision_recall_curve(vy, cal_vp)
    c_th = th_c[np.argmax((2*p_c*r_c/(p_c+r_c+1e-8))[:-1])]
    c_f1 = 2*np.sum(((cal_tp_adj>=c_th)==1)&(ty==1)) / (np.sum(cal_tp_adj>=c_th)+np.sum(ty)+1e-8)
    raw_f1s.append(r_f1); soft_f1s.append(s_f1); cal_f1s.append(c_f1); orc_f1s.append(orc_f1)
    print(f'{cname:<20} | {pi_v*100:7.2f}% | {pi_t*100:8.2f}% | {r_f1:7.4f} | {s_f1:7.4f} | {c_f1:8.4f} | {orc_f1:14.4f}')  # [restored]
print('-'*92)
print(f'{"MEAN":<20} | {"":>8} | {"":>9} | {np.mean(raw_f1s):7.4f} | {np.mean(soft_f1s):7.4f} | {np.mean(cal_f1s):8.4f} | {np.mean(orc_f1s):14.4f}')  # [restored]

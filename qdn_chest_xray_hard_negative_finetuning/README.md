# QueryDiscussNet (QDN) for chest X-ray classification: gate, soft-gating and hard-negative fine-tuning

**Aim:** raise the natural-prevalence F1 and AUPRC of a pretrained QueryDiscussNet (QDN) model on three focus findings (Effusion, Pneumothorax, Cardiomegaly), and reduce false positives on scans that carry other findings.
**Model:** QDN with a timm `resnet50d` backbone, 256-dim patch tokens (C4 stride 16 and C5 stride 32) with 2D sine-cosine position encoding, 12 learned "doctor" queries (11 pathologies + `Is_Abnormal`), 3 shared read / talk / write steps and 4 learned prototypes per class.
**Tools:** PyTorch, timm, scikit-learn, NumPy, pandas; 4-GPU server.

## Procedure
1. **Stage-1 gate and thresholds** (`scripts/evaluate_optimized.py`): the 12th output (`Is_Abnormal`) is a gate; a scan with gate probability below 0.30 gets its pathology probabilities zeroed. Per-class thresholds are swept on the validation split only and applied to the test split.
2. **False-positive forensics** (`scripts/inspect_top_fps.py`): list the most confident false positives with view position (PA/AP), abnormality flag and true findings. 7,029 unique false-positive images were exported to `hard_negatives_to_mine.txt`.
3. **Fine-tuning fixes.** Problems met while moving from post-processing to fine-tuning, and how each was resolved:

| Problem | Cause | Fix |
|---|---|---|
| State-dict shape mismatch (`proj4.weight` …) | ad-hoc torchvision ResNet-50 wrapper loaded with `strict=False`; checkpoint used timm `resnet50d`, `embed_dim=256`, 3 shared steps, 4 prototypes | import the native `QueryDiscussNet` and load `best_ema.pt` with `strict=True` |
| DataLoader `FileNotFoundError` | loader looked in `./images`, which did not exist | use the pre-extracted 224×224 image cache through the native `ChestXrayDataset` and `configs/top3_seed42.yaml` |
| NaN loss around batch 655 | BCE inside the FP16 autocast block overflowed; skipped batches still updated BatchNorm statistics | compute the loss outside autocast on `logits.float()`, as in `training/train_pipeline.py` |
| Pretrained EMA destroyed on step 1 | `ModelEMA` starts at `updates = 0`; warm-up decay `(1+u)/(10+u)` = 0.18 on step 1 overwrites 82 % of the checkpoint | set `ema.updates = 5000` and `decay = 0.998` right after loading |
| Test false-positive list never matches train | patient-level splits are disjoint, so test filenames are never in `train.csv` | mine confounders on the training split from labels (`yc = 0` and `yabn = 1`) |

4. **Hard-negative suppression fine-tuning** (`scripts/hard_negative_loss.py`): loss weight for a focus class `c` is `wc = 1 + 1[yc=0] · α_FP · pc^γ · (1 + 0.5·yabn)` with `α_FP = 3.5`, `γ = 2.0`, computed in FP32. `WeightedRandomSampler` over 77,238 training scans: abnormal scans with no focus finding at 2.0×, positive focus scans at 2.5×. AdamW with backbone LR = 0.15 × base and QDN heads at 4.0× the backbone rate, 2 epochs, cosine annealing, gradient clipping at 2.0. Outputs go to `outputs/final/qdn_hard_neg/`.
5. **384×384 run** (`outputs/final/qdn_384/`): 4-epoch training on 111,821 cached 384×384 images, then evaluation. Predictions for the gate and the 11 `cls_*` heads are stored as `val/test_probs.npy` and `val/test_labels.npy`, and all later analyses work from these arrays: soft-gating `p_cls · p_gate^α` (α in [0, 1.5] and the threshold tuned on validation only), false-positive stratification, prevalence-shift check, and a validation-locked 1:1 case-control benchmark.

## Results
**Phase 1: gate τ = 0.30 + validation-tuned thresholds, baseline QDN (test F1, from the report):** Effusion 0.5312, Pneumothorax 0.4695, Cardiomegaly 0.4187.

**Hard-negative fine-tuning (2-epoch run):** the report's results table for this run was never filled in (the cells read "Awaiting Run Output"), so no fine-tuned numbers exist in the source.

**384×384 run:**
- Training: 3,218 steps per epoch (448 s); loss 0.2210 at epoch 3 and 0.2171 at epoch 4. Validation focus mean AUPRC peaked at epoch 3 (0.3242: Effusion 0.541, Pneumothorax 0.213, Cardiomegaly 0.219; epoch 4: 0.3227) and the epoch-3 weights were kept as `best_ema.pt`.
- One image (`00021618_015.png`) failed with `[Errno 22]` while caching (111,820 of 111,821 cached).
- Test set: 25,536 scans, about 9,847 with no abnormality.

Baseline (raw) metrics per module:

| module | val auprc | test auroc | test auprc | val threshold |
|---|---|---|---|---|
| gate (abnormality) | 0.7026 | 0.7322 | 0.7846 | 0.53 |
| Atelectasis | 0.3682 | 0.7762 | 0.3566 | 0.22 |
| Cardiomegaly | 0.2187 | 0.8804 | 0.3302 | 0.26 |
| Consolidation | 0.1382 | 0.7421 | 0.1621 | 0.18 |
| Edema | 0.1581 | 0.8462 | 0.1680 | 0.23 |
| Effusion | 0.5406 | 0.8297 | 0.5212 | 0.29 |
| Emphysema | 0.2855 | 0.9220 | 0.4506 | 0.31 |
| Infiltration | 0.3250 | 0.6907 | 0.3902 | 0.32 |
| Mass | 0.3027 | 0.8221 | 0.3195 | 0.31 |
| Nodule | 0.2668 | 0.7804 | 0.2572 | 0.30 |
| Pleural_Thickening | 0.0946 | 0.7622 | 0.1246 | 0.13 |
| Pneumothorax | 0.2132 | 0.8641 | 0.4161 | 0.17 |
| 11-class mean | 0.2647 | 0.8106 | 0.3179 | – |

Raw vs soft-gated (`p_cls · p_gate^α`, α and threshold tuned on validation). Hard gating at 0.30 changed nothing for Effusion because every scan with gate < 0.30 already had an Effusion probability ≤ 0.1533, below the 0.29 threshold. Soft gating raised mean test AUPRC from 0.3179 to 0.3201 and mean test F1 from 0.3716 to 0.3762 (the soft-gated F1 column is cut off in the source tables):

| class | alpha | raw auprc | soft auprc | raw f1 |
|---|---|---|---|---|
| Atelectasis | 0.50 | 0.3566 | 0.3589 | 0.3987 |
| Cardiomegaly | 0.30 | 0.3302 | 0.3312 | 0.3904 |
| Consolidation | 0.30 | 0.1621 | 0.1617 | 0.2174 |
| Edema | 1.30 | 0.1680 | 0.1692 | 0.2292 |
| Effusion | 0.20 | 0.5212 | 0.5216 | 0.5288 |
| Emphysema | 1.20 | 0.4506 | 0.4589 | 0.5206 |
| Infiltration | 0.50 | 0.3902 | 0.3989 | 0.4573 |
| Mass | 0.10 | 0.3195 | 0.3200 | 0.3710 |
| Nodule | 0.50 | 0.2572 | 0.2591 | 0.2965 |
| Pleural_Thickening | 0.20 | 0.1246 | 0.1249 | 0.1940 |
| Pneumothorax | 0.10 | 0.4161 | 0.4164 | 0.4833 |
| mean | – | 0.3179 | 0.3201 | 0.3716 |

Natural-prevalence F1 and other test metrics after soft gating (sensitivity at the Youden operating point):

| class | test auroc | sensitivity youden | test accuracy | natural f1 val locked |
|---|---|---|---|---|
| Atelectasis | 0.7777 | 0.8344 | 0.7765 | 0.4005 |
| Cardiomegaly | 0.8807 | 0.8527 | 0.9524 | 0.4042 |
| Consolidation | 0.7421 | 0.8616 | 0.8644 | 0.2134 |
| Edema | 0.8464 | 0.9252 | 0.9307 | 0.2350 |
| Effusion | 0.8299 | 0.9049 | 0.7846 | 0.5284 |
| Emphysema | 0.9218 | 0.8836 | 0.9549 | 0.5237 |
| Infiltration | 0.6969 | 0.7714 | 0.6216 | 0.4634 |
| Mass | 0.8209 | 0.7710 | 0.9165 | 0.3654 |
| Nodule | 0.7800 | 0.6471 | 0.9210 | 0.3183 |
| Pleural_Thickening | 0.7625 | 0.8374 | 0.8923 | 0.2013 |
| Pneumothorax | 0.8651 | 0.8731 | 0.8719 | 0.4851 |
| 11-class mean | 0.8113 | 0.8330 | 0.8624 | 0.3762 |

Summary across evaluation protocols (means over the 11 classes):

| Protocol | Test AUROC | Test AUPRC | Test F1 |
|---|---|---|---|
| Raw, natural prevalence | 0.8106 | 0.3179 | 0.3716 |
| Soft-gated, natural prevalence (validation-locked) | 0.8113 | 0.3201 | 0.3762 |
| Oracle threshold on the test set (upper bound) | 0.8113 | 0.3201 | 0.3835 |
| 1:1 case-control (validation-locked) | 0.8113 | 0.7940 | 0.7515 |
| 1:1 case-control vs healthy scans only (validation-locked) | 0.8362 | 0.8145 | 0.7755 |

Where the false positives are (soft-gated heads, test split):

| class | total soft fp | fp on normal lungs | pct normal | fp on other pathologies | pct other | specificity on normal lungs pct |
|---|---|---|---|---|---|---|
| Consolidation | 2046 | 529 | 25.9 | 1517 | 74.1 | 94.6 |
| Edema | 1497 | 475 | 31.7 | 1022 | 68.3 | 95.2 |
| Pleural_Thickening | 1625 | 544 | 33.5 | 1081 | 66.5 | 94.5 |
| Nodule | 1793 | 635 | 35.4 | 1158 | 64.6 | 93.6 |
| Mass | 1311 | 493 | 37.6 | 818 | 62.4 | 95.0 |
| Atelectasis | 4385 | 1705 | 38.9 | 2680 | 61.1 | 82.7 |
| Effusion | 4116 | 1604 | 39.0 | 2512 | 61.0 | 83.7 |
| Emphysema | 730 | 298 | 40.8 | 432 | 59.2 | 97.0 |
| Pneumothorax | 2118 | 953 | 45.0 | 1165 | 55.0 | 90.3 |
| Infiltration | 8897 | 4431 | 49.8 | 4466 | 50.2 | 55.0 |
| Cardiomegaly | 616 | 328 | 53.2 | 288 | 46.8 | 96.7 |

Positive prevalence, validation vs test:

| class | val prevalence pct | test prevalence pct | shift ratio |
|---|---|---|---|
| Atelectasis | 9.58 | 12.81 | 1.34 |
| Cardiomegaly | 1.96 | 4.17 | 2.13 |
| Consolidation | 3.37 | 7.07 | 2.10 |
| Edema | 1.32 | 3.61 | 2.73 |
| Effusion | 10.75 | 18.21 | 1.69 |
| Emphysema | 1.59 | 4.27 | 2.69 |
| Infiltration | 16.27 | 23.88 | 1.47 |
| Mass | 4.63 | 6.84 | 1.48 |
| Nodule | 5.88 | 6.35 | 1.08 |
| Pleural_Thickening | 2.68 | 4.46 | 1.66 |
| Pneumothorax | 2.89 | 10.40 | 3.60 |
| mean | 5.54 | 9.28 | 1.68 |

PA vs AP view, focus classes:

| class | pa auroc | pa auprc | ap auroc | ap auprc |
|---|---|---|---|---|
| Cardiomegaly | 0.9338 | 0.4234 | 0.8310 | 0.2586 |
| Effusion | 0.8642 | 0.5620 | 0.8015 | 0.4934 |
| Pneumothorax | 0.8623 | 0.4614 | 0.8526 | 0.3384 |

## Interpretation
- **False positives are mostly cross-talk, not healthy-lung alarms.** For Consolidation, Edema, Pleural Thickening, Nodule, Mass, Atelectasis and Effusion, 61–74 % of false positives fall on abnormal scans that carry some other finding. This is also why the Stage-1 gate cannot remove them: those scans really are abnormal, so suppressing them has to happen inside the classifier (the motivation for the hard-negative loss). The weakest head on normal lungs is Infiltration (55.0 % specificity); Cardiomegaly and Emphysema false-alarm on only about 3 % of normal scans.
- **Thresholding is not the bottleneck.** The soft-gated F1 (0.3762) is within 0.0073 of the best possible F1 on the test set for these probabilities (0.3835), so further threshold tuning cannot lift natural-prevalence F1 much; improvements have to come from the model.
- **Validation-to-test prevalence shift.** Test prevalence is higher than validation for every class (1.08× to 3.60×; Pneumothorax 2.89 % to 10.40 %), so thresholds tuned on validation are calibrated for a lower-prevalence population.
- **Why natural F1 looks low while AUROC is high.** With about 25 negatives per positive for rare classes (e.g. Cardiomegaly 4.17 %), F1 is dominated by false positives even at good sensitivity and specificity. The 1:1 case-control F1 (0.7515) is a different protocol: it removes the class imbalance and is not comparable to the natural-prevalence F1, so report the two side by side and label them.
- **View check.** Cardiomegaly separates better on erect PA views (AUROC 0.9338) than on supine AP views (0.8310); Effusion and Pneumothorax stay discriminative on both views, with higher AUPRC on PA.
- **Best classes:** Emphysema (AUROC 0.9220), Cardiomegaly (0.8804), Pneumothorax (0.8641). **Hardest:** Infiltration (AUROC 0.6907), Consolidation (AUPRC 0.1621) and Pleural Thickening (AUPRC 0.1246).

## Notes and gaps

- **Two different runs.** The Phase 1 numbers (Effusion 0.5312 etc.) and the 384×384 numbers (Effusion 0.5284) come from different runs, so they are not directly comparable.



## Files
```
scripts/
  evaluate_optimized.py  inspect_top_fps.py  hard_negative_loss.py     (from the report)
  analysis_384/                                                         (from the second chat, 384x384 run)
    01_baseline_metrics.py                 raw AUROC/AUPRC/F1 per module, threshold tuned on validation
    02_soft_gating_export.py               hard vs soft gate, alpha sweep, writes full_11class_metrics.csv/json
    03_false_positive_stratification.py    FPs on normal lungs vs other pathologies
    04_prevalence_shift_oracle_calibration.py  val/test prevalence, oracle F1 ceiling, logistic calibration
    05_case_control_benchmark.py           Youden sensitivity/specificity, 1:1 case-control F1 (own label, not comparable to natural F1)
  early_attempts/                                                       (from the first chat, debugging history)
    dataset_fixed.py  find_images_dir.py  update_img_dir.py  update_finetune_path.py
    fix_nan_finetune.py  finetune_qdn_adhoc.py
results/   CSV tables recovered from the report and chats
reports/   QDN_optimization_report.pdf
```


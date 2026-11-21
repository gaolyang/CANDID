# CANDID — Supplementary Experimental Results

> Source: Appendix R-A, R-B and R-C of the earlier response letter (`CANDID_ResponseLetter-v12(7).docx`). The numerical entries below are transcribed from the original tables without recalculation or modification.

This document provides the full task-order results and the run-level/statistical details corresponding to the CANDID manuscript. The task-order experiments and the five-seed, fixed-task-order statistical analysis are **separate experimental protocols** and must not be pooled together.

## Contents

- [S1. Detailed task-order robustness results (Appendix R-A)](#s1-detailed-task-order-robustness-results-appendix-r-a)
- [S2. Run-level values underlying repeated-run statistics (Appendix R-B)](#s2-run-level-values-underlying-repeated-run-statistics-appendix-r-b)
- [S3. Paired statistical comparisons with CANDID (Appendix R-C)](#s3-paired-statistical-comparisons-with-candid-appendix-r-c)

---

## S1. Detailed task-order robustness results (Appendix R-A)

This appendix reports the complete task-wise task-order robustness results referenced in the responses to Reviewers #2 and #3. For each dataset, three attack-arrival orders are evaluated under the Global, task-aware, and old-attack protocols. These task-order runs vary the attack-arrival sequence and are separate from the fixed-order five-seed repeated-run analysis; they are not included in the mean ± SD calculations, Wilcoxon tests, or Holm correction.

### A1  CICIDS2018 - Order A

**Task sequence: Base → Web Attack → BruteForce → Bot**

#### Table A1. CICIDS2018 global evaluation under task order A

| Task | New Task | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 96.50 | 92.42 | 79.48 | 85.46 | 0.97 |
| T1 | Web Attack | 96.92 | 90.22 | 85.41 | 87.74 | 1.38 |
| T2 | BruteForce | 97.97 | 97.92 | 86.17 | 91.67 | 0.27 |
| T3 | Bot | 98.89 | 96.84 | 94.52 | 95.66 | 0.46 |

#### Table A2. CICIDS2018 task-aware evaluation under task order A

| Task | Seen Attack Set | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 97.45 | 98.76 | 92.46 | 95.51 | 0.48 |
| T1 | Base + Web Attack | 97.30 | 98.33 | 92.36 | 95.25 | 0.65 |
| T2 | Base + Web Attack + BruteForce | 97.27 | 98.57 | 93.64 | 96.04 | 0.74 |
| T3 | All Attacks | 97.81 | 99.58 | 94.52 | 96.98 | 0.24 |

#### Table A3. CICIDS2018 old-attack evaluation under task order A

| Task | Old Attack Set | Acc (%) | Precision (%) | Old Recall (RcOld) | Old F1 | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | N/A | N/A | N/A | N/A | N/A | N/A |
| T1 | Base attacks | 97.30 | 98.33 | 92.35 | 95.25 | 0.65 |
| T2 | Base + Web Attack | 97.02 | 98.08 | 91.63 | 94.75 | 0.74 |
| T3 | Base + Web Attack + BruteForce | 97.74 | 99.54 | 94.04 | 96.72 | 0.24 |

### A2  CICIDS2018 - Order B

**Task sequence: Base → BruteForce → Bot → Web Attack**

#### Table A4. CICIDS2018 global evaluation under task order B

| Task | New Task | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 96.41 | 91.68 | 79.47 | 85.14 | 1.07 |
| T1 | BruteForce | 98.03 | 98.28 | 86.26 | 91.88 | 0.22 |
| T2 | Bot | 98.99 | 97.84 | 94.29 | 96.04 | 0.31 |
| T3 | Web Attack | 98.60 | 94.56 | 94.65 | 94.61 | 0.81 |

#### Table A5. CICIDS2018 task-aware evaluation under task order B

| Task | Seen Attack Set | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 97.41 | 98.65 | 92.43 | 95.44 | 0.53 |
| T1 | Base + BruteForce | 97.73 | 99.77 | 93.77 | 96.68 | 0.12 |
| T2 | Base + BruteForce + Bot | 97.42 | 98.65 | 94.36 | 96.46 | 0.77 |
| T3 | All Attacks | 97.65 | 98.99 | 94.65 | 96.77 | 0.57 |

#### Table A6. CICIDS2018 old-attack evaluation under task order B

| Task | Old Attack Set | Acc (%) | Precision (%) | Old Recall (RcOld) | Old F1 | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | N/A | N/A | N/A | N/A | N/A | N/A |
| T1 | Base attacks | 97.51 | 99.69 | 91.80 | 95.58 | 0.12 |
| T2 | Base + BruteForce | 97.34 | 98.52 | 93.88 | 96.15 | 0.77 |
| T3 | Base + BruteForce + Bot | 97.65 | 98.99 | 94.65 | 96.77 | 0.57 |

### A3  CICIDS2018 - Order C

**Task sequence: Base → Bot → Web Attack → BruteForce**

#### Table A7. CICIDS2018 global evaluation under task order C

| Task | New Task | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 94.56 | 78.65 | 79.52 | 79.08 | 3.21 |
| T1 | Bot | 98.02 | 97.35 | 87.06 | 91.92 | 0.35 |
| T2 | Web Attack | 98.38 | 92.91 | 94.72 | 93.81 | 1.07 |
| T3 | BruteForce | 98.95 | 97.44 | 94.34 | 95.87 | 0.37 |

#### Table A8. CICIDS2018 task-aware evaluation under task order C

| Task | Seen Attack Set | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 96.77 | 96.33 | 92.52 | 94.39 | 1.46 |
| T1 | Base + Bot | 97.56 | 99.51 | 92.73 | 96.00 | 0.21 |
| T2 | Base + Bot + Web Attack | 96.72 | 96.29 | 93.23 | 94.73 | 1.66 |
| T3 | All Attacks | 97.80 | 99.75 | 94.34 | 96.97 | 0.14 |

#### Table A9. CICIDS2018 old-attack evaluation under task order C

| Task | Old Attack Set | Acc (%) | Precision (%) | Old Recall (RcOld) | Old F1 | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | N/A | N/A | N/A | N/A | N/A | N/A |
| T1 | Base attacks | 97.48 | 99.45 | 91.90 | 95.53 | 0.21 |
| T2 | Base + Bot | 96.72 | 96.28 | 93.23 | 94.73 | 1.66 |
| T3 | Base + Bot + Web Attack | 97.61 | 99.67 | 92.74 | 96.08 | 0.14 |

### A4  UNSW-NB15 - Order A

**Task sequence: Base → Worms → Exploits → Generic**

#### Table A10. UNSW-NB15 global evaluation under task order A

| Task | New Task | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 99.94 | 98.98 | 99.98 | 99.48 | 0.06 |
| T1 | Worms | 99.97 | 99.70 | 99.74 | 99.72 | 0.02 |
| T2 | Exploits | 99.97 | 99.70 | 99.83 | 99.76 | 0.02 |
| T3 | Generic | 99.96 | 99.50 | 99.83 | 99.67 | 0.03 |

#### Table A11. UNSW-NB15 task-aware evaluation under task order A

| Task | Seen Attack Set | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 99.97 | 99.76 | 100.00 | 99.88 | 0.03 |
| T1 | Base + Worms | 99.96 | 99.95 | 99.70 | 99.83 | 0.01 |
| T2 | Base + Worms + Exploits | 99.94 | 99.84 | 99.81 | 99.83 | 0.03 |
| T3 | All Attacks | 99.94 | 99.86 | 99.83 | 99.85 | 0.03 |

#### Table A12. UNSW-NB15 old-attack evaluation under task order A

| Task | Old Attack Set | Acc (%) | Precision (%) | Old Recall (RcOld) | Old F1 | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | Base attacks | 99.96 | 99.95 | 99.70 | 99.83 | 0.01 |
| T2 | Base + Worms | 99.95 | 99.74 | 99.74 | 99.74 | 0.03 |
| T3 | Base + Worms + Exploits | 99.94 | 99.84 | 99.80 | 99.82 | 0.03 |

### A5  UNSW-NB15 - Order B

**Task sequence: Base → Exploits → Generic → Worms**

#### Table A13. UNSW-NB15 global evaluation under task order B

| Task | New Task | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 99.96 | 99.33 | 99.96 | 99.64 | 0.04 |
| T1 | Exploits | 99.98 | 99.63 | 99.93 | 99.78 | 0.02 |
| T2 | Generic | 99.97 | 99.92 | 99.60 | 99.76 | 0.00 |
| T3 | Worms | 99.93 | 99.93 | 98.71 | 99.32 | 0.00 |

#### Table A14. UNSW-NB15 task-aware evaluation under task order B

| Task | Seen Attack Set | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 99.98 | 99.78 | 100.00 | 99.89 | 0.03 |
| T1 | Base + Exploits | 99.98 | 99.97 | 99.93 | 99.95 | 0.01 |
| T2 | Base + Exploits + Generic | 99.92 | 99.96 | 99.60 | 99.78 | 0.01 |
| T3 | All Attacks | 99.75 | 99.96 | 98.71 | 99.33 | 0.01 |

#### Table A15. UNSW-NB15 old-attack evaluation under task order B

| Task | Old Attack Set | Acc (%) | Precision (%) | Old Recall (RcOld) | Old F1 | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | Base attacks | 99.99 | 99.95 | 99.94 | 99.94 | 0.01 |
| T2 | Base + Exploits | 99.92 | 99.95 | 99.55 | 99.75 | 0.01 |
| T3 | Base + Exploits + Generic | 99.75 | 99.96 | 98.71 | 99.33 | 0.01 |

### A6  UNSW-NB15 - Order C

**Task sequence: Base → Generic → Worms → Exploits**

#### Table A16. UNSW-NB15 global evaluation under task order C

| Task | New Task | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 99.96 | 99.33 | 99.91 | 99.62 | 0.04 |
| T1 | Generic | 99.93 | 98.88 | 99.82 | 99.35 | 0.06 |
| T2 | Worms | 99.96 | 99.91 | 99.33 | 99.62 | 0.01 |
| T3 | Exploits | 99.95 | 99.11 | 99.99 | 99.55 | 0.05 |

#### Table A17. UNSW-NB15 task-aware evaluation under task order C

| Task | Seen Attack Set | Acc (%) | Precision (%) | Recall (%) | F1 (%) | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Base | 99.97 | 99.80 | 99.95 | 99.87 | 0.02 |
| T1 | Base + Generic | 99.98 | 99.89 | 99.95 | 99.92 | 0.02 |
| T2 | Base + Generic + Worms | 99.97 | 99.95 | 99.83 | 99.89 | 0.01 |
| T3 | All Attacks | 99.95 | 99.73 | 99.99 | 99.86 | 0.06 |

#### Table A18. UNSW-NB15 old-attack evaluation under task order C

| Task | Old Attack Set | Acc (%) | Precision (%) | Old Recall (RcOld) | Old F1 | FPR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | Base attacks | 99.98 | 99.89 | 99.95 | 99.92 | 0.02 |
| T2 | Base + Generic | 99.97 | 99.95 | 99.83 | 99.89 | 0.01 |
| T3 | Base + Generic + Worms | 99.95 | 99.73 | 99.99 | 99.86 | 0.06 |

Together, Tables A1–A18 provide the full task-wise evidence supporting the concise task-order robustness statement added to Section 4.2.5 of the revised manuscript. The Tukey-IQR threshold rationale is provided in Section 3.3.3 (Reference 41 of the manuscript), uncertainty-bound and replay-budget sensitivity are reported in Tables 9–10, and the naturally varying benign/attack imbalance is documented in Section 4.1 and Table 2.

---

## S2. Run-level values underlying repeated-run statistics (Appendix R-B)

This appendix reports the matched final-task F1Seen and RcOld values used in the repeated-run statistical analysis. For each method, the five values in every cell follow the fixed seed order 42 / 52 / 62 / 72 / 82. The task order and the remaining experimental settings are fixed across these repeated runs, and sample standard deviation in the manuscript uses N = 5 and ddof = 1.

Only final-task F1Seen and RcOld are tabulated here. The paired statistical comparisons derived from these matched runs are reported in Appendix R-C. The separate task-order robustness experiments in Appendix R-A vary the attack-arrival sequence and are excluded from both the mean ± SD calculations and the paired statistical tests.

### B1  CICIDS2018 final-task run-level results

| Method | F1Seen (%) Seeds 42 / 52 / 62 / 72 / 82 | RcOld (%) Seeds 42 / 52 / 62 / 72 / 82 |
| --- | --- | --- |
| JT | 97.52 / 97.39 / 97.47 / 97.40 / 97.26 | 95.30 / 95.28 / 95.40 / 95.34 / 95.08 |
| EWC | 90.77 / 88.79 / 89.07 / 89.31 / 88.16 | 82.68 / 83.20 / 83.80 / 84.10 / 82.40 |
| LUCIR | 90.70 / 90.85 / 91.10 / 90.64 / 90.35 | 84.00 / 84.20 / 84.60 / 83.90 / 83.50 |
| ECBRS | 83.28 / 83.72 / 82.93 / 83.54 / 82.55 | 71.80 / 72.40 / 71.20 / 72.10 / 70.80 |
| TAMR | 87.13 / 87.29 / 87.03 / 87.22 / 86.80 | 78.60 / 78.80 / 78.45 / 78.70 / 78.15 |
| CANDID | 96.98 / 96.90 / 96.90 / 96.79 / 96.87 | 94.04 / 93.79 / 93.68 / 93.56 / 93.69 |

### B2  UNSW-NB15 final-task run-level results

| Method | F1Seen (%) Seeds 42 / 52 / 62 / 72 / 82 | RcOld (%) Seeds 42 / 52 / 62 / 72 / 82 |
| --- | --- | --- |
| JT | 97.48 / 97.45 / 97.58 / 97.35 / 97.25 | 95.50 / 95.45 / 95.60 / 95.28 / 95.15 |
| EWC | 89.58 / 89.65 / 89.46 / 89.70 / 88.99 | 98.10 / 98.05 / 98.20 / 97.90 / 97.80 |
| LUCIR | 94.69 / 94.30 / 94.57 / 94.07 / 93.80 | 92.35 / 94.20 / 94.50 / 93.90 / 93.60 |
| ECBRS | 92.08 / 91.98 / 92.30 / 91.69 / 91.28 | 86.50 / 89.50 / 90.00 / 89.20 / 88.65 |
| TAMR | 92.44 / 92.51 / 92.25 / 92.43 / 92.04 | 86.12 / 87.40 / 87.00 / 87.25 / 86.70 |
| CANDID | 99.85 / 99.77 / 99.90 / 99.79 / 99.91 | 99.80 / 99.85 / 99.98 / 99.72 / 99.90 |

---

## S3. Paired statistical comparisons with CANDID (Appendix R-C)

For final-task F1Seen and RcOld, CANDID is compared with JT, EWC, LUCIR, ECBRS, and TAMR using two-sided exact Wilcoxon signed-rank tests on the five matched random-seed labels (42, 52, 62, 72, and 82). Positive mean differences and positive rank-biserial correlations favor CANDID. Holm adjustment is applied across the ten method-by-metric comparisons within each dataset; N = 5 for both datasets.

With five matched non-zero pairs, the two-sided exact Wilcoxon p-values are discrete; when all five paired differences have the same sign, the smallest attainable two-sided p-value is 0.0625. This explains the repeated raw p = 0.0625 values below and underscores the limited inferential power of N = 5.

### C1  CICIDS2018 final-task F1Seen and RcOld

| Metric | CANDID vs. | N | Mean difference (pp) | Rank-biserial r | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| F1Seen | JT | 5 | -0.52 | -1.00 | 0.0625 | 0.6250 |
| F1Seen | EWC | 5 | 7.67 | 1.00 | 0.0625 | 0.6250 |
| F1Seen | LUCIR | 5 | 6.16 | 1.00 | 0.0625 | 0.6250 |
| F1Seen | ECBRS | 5 | 13.68 | 1.00 | 0.0625 | 0.6250 |
| F1Seen | TAMR | 5 | 9.79 | 1.00 | 0.0625 | 0.6250 |
| RcOld | JT | 5 | -1.53 | -1.00 | 0.0625 | 0.6250 |
| RcOld | EWC | 5 | 10.52 | 1.00 | 0.0625 | 0.6250 |
| RcOld | LUCIR | 5 | 9.71 | 1.00 | 0.0625 | 0.6250 |
| RcOld | ECBRS | 5 | 22.09 | 1.00 | 0.0625 | 0.6250 |
| RcOld | TAMR | 5 | 15.21 | 1.00 | 0.0625 | 0.6250 |

No comparison reaches p < 0.05 after Holm adjustment. Every F1Seen and RcOld comparison has raw p = 0.0625 and Holm-adjusted p = 0.6250. The paired effect sizes are descriptive, and the five-run sample is too small for a strong inferential claim.

### C2  UNSW-NB15 final-task F1Seen and RcOld

| Metric | CANDID vs. | N | Mean difference (pp) | Rank-biserial r | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| F1Seen | JT | 5 | 2.42 | 1.00 | 0.0625 | 0.6250 |
| F1Seen | EWC | 5 | 10.37 | 1.00 | 0.0625 | 0.6250 |
| F1Seen | LUCIR | 5 | 5.56 | 1.00 | 0.0625 | 0.6250 |
| F1Seen | ECBRS | 5 | 7.98 | 1.00 | 0.0625 | 0.6250 |
| F1Seen | TAMR | 5 | 7.51 | 1.00 | 0.0625 | 0.6250 |
| RcOld | JT | 5 | 4.45 | 1.00 | 0.0625 | 0.6250 |
| RcOld | EWC | 5 | 1.84 | 1.00 | 0.0625 | 0.6250 |
| RcOld | LUCIR | 5 | 6.14 | 1.00 | 0.0625 | 0.6250 |
| RcOld | ECBRS | 5 | 11.08 | 1.00 | 0.0625 | 0.6250 |
| RcOld | TAMR | 5 | 12.96 | 1.00 | 0.0625 | 0.6250 |

No comparison reaches p < 0.05 after Holm adjustment. Every F1Seen and RcOld comparison has raw p = 0.0625 and Holm-adjusted p = 0.6250. The paired effect sizes are descriptive, and the five-run sample is too small for a strong inferential claim.

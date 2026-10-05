# Lightweight Wavelet-SVM for MARIDA Maritime Anomaly Detection

This branch contains the leakage-controlled experimental pipeline used for the manuscript.

## Locked methodology

MARIDA water patches are class 0 (normal). Marine-debris and ship patches are class 1 (maritime anomaly). The natural eligible class distribution is retained; no global undersampling is performed.

The official MARIDA train + validation splits form the development set. The official test split is sealed during representation, classifier, and hyperparameter selection. Preprocessing parameters are fitted only from development/training folds.

The development study selected NIR 842 nm (Sentinel-2 channel 8 in the MARIDA TIFF) -> db4, two-level DWT -> 18 statistics -> signed-log transform -> z-score -> RBF-SVM (C=10, KernelScale=1).

Step 8 performs spectral ablation using development CV only. Step 8B uses repeated stratified 5-fold development CV for a close tie. The final configuration was then evaluated once by Step 9 on the official test split.

## Reproducible workflow

Run MATLAB from the code/ directory. run_all prepares the natural-distribution dataset and development split only. It intentionally does not evaluate the official test set.

Workflow: run_all; step8_spectral_ablation_dev; step8b_tiebreak_red_nir_dev; lock configuration; step9_final_locked_test exactly once; step7_compare_baselines for development-only classifier context; step6_benchmark for runtime/model-size measurement.

Do not rerun Step 9 to choose another representation, classifier, threshold, or hyperparameter.

## Final locked official-test result

Development: N=715 (328 normal, 387 anomaly). Official test: N=257 (111 normal, 146 anomaly).

Locked NIR-842 RBF-SVM: Accuracy 71.5953%; Balanced accuracy 70.1407%; Precision 0.723926; Recall 0.808219; Specificity 0.594595; F1 0.763754; ROC-AUC 0.759102; confusion matrix TN=66, FP=45, FN=28, TP=118.

The relatively high recall and lower specificity indicate a sensitivity/false-positive trade-off that should be reported explicitly.

## Key files

results/spectral_ablation_dev.* — development-only representation study.
results/spectral_tiebreak_dev.* — repeated-CV selection record.
results/final_locked_test.* — final official-test evaluation.
models/final_locked_svm.mat — locked deployment model.
results/tiff_audit.txt — audit confirming 11-band MARIDA TIFF input.

Historical experiments remain available through Git history but are intentionally removed from the branch working tree when they could be confused with the locked protocol.

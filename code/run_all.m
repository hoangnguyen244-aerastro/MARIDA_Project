%% run_all.m
% SAFE DEVELOPMENT PIPELINE. Official test is intentionally NOT evaluated.
% Run step9_final_locked_test ONLY once after development selection is locked.
function run_all()
fprintf('\n========================================================\n');
fprintf(' MARIDA DEVELOPMENT PIPELINE - TEST SET SEALED\n');
fprintf('========================================================\n');
tic;
step1_filter_data();
if ~exist('../data/file_lists.mat','file'), error('Step 1 failed.'); end
step2_extract_features();
if ~exist('../data/features_data.mat','file'), error('Step 2 failed.'); end
step3_train_svm();
if ~exist('../models/svm_model.mat','file'), error('Step 3 failed.'); end
fprintf('\nDevelopment preparation complete.\n');
fprintf('Next: run step8_spectral_ablation_dev, then step8b_tiebreak_red_nir_dev.\n');
fprintf('DO NOT run Step 9 unless the development configuration is locked.\n');
fprintf('Elapsed %.2f s.\n',toc);
end
%% run_all.m
% Master script to run the entire pipeline
% Run this file to execute all steps sequentially
% Compatible with MATLAB R2015b

function run_all()
    
    fprintf('\n');
    fprintf('========================================================\n');
    fprintf('   WAVELET-SVM ANOMALY DETECTION FOR SATELLITE IMAGERY   \n');
    fprintf('                   Complete Pipeline                     \n');
    fprintf('========================================================\n');
    
    tic;
    
    % Step 1: Filter and balance data
    fprintf('\n');
    step1_filter_data();
    
    % Check if step 1 succeeded (check in data folder)
    if ~exist('../data/file_lists.mat', 'file')
        fprintf('\nERROR: Pipeline stopped at Step 1\n');
        return;
    end
    
    % Step 2: Extract wavelet features
    fprintf('\n');
    step2_extract_features();
    
    % Check if step 2 succeeded
    if ~exist('../data/features_data.mat', 'file')
        fprintf('\nERROR: Pipeline stopped at Step 2\n');
        return;
    end
    
    % Step 3: Train SVM
    fprintf('\n');
    step3_train_svm();
    
    % Check if step 3 succeeded
    if ~exist('../models/svm_model.mat', 'file')
        fprintf('\nERROR: Pipeline stopped at Step 3\n');
        return;
    end
    
    % Step 4: Evaluate
    fprintf('\n');
    step4_evaluate();
    
    % Step 5: Compare wavelets (uncomment if desired)
    fprintf('\n');
    fprintf('=== Running Step 5: Wavelet Comparison ===\n');
    step5_compare_wavelets();
    
    elapsed_time = toc;
    
    fprintf('\n========================================================\n');
    fprintf('PIPELINE COMPLETED SUCCESSFULLY!\n');
    fprintf('Total execution time: %.2f seconds\n', elapsed_time);
    fprintf('========================================================\n');
    
end
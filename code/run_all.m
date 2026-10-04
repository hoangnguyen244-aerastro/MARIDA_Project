%% run_all.m
% Master script - Wavelet-SVM Anomaly Detection (v5 - official MARIDA split)
% Compatible with MATLAB R2015b

function run_all()
    
    fprintf('\n');
    fprintf('========================================================\n');
    fprintf('   WAVELET-SVM ANOMALY DETECTION - FINAL PIPELINE      \n');
    fprintf('       (18 features, official MARIDA split)                            \n');
    fprintf('========================================================\n');
    
    tic;
    
    % Step 1: Filter data (FULL MARIDA)
    step1_filter_data();
    if ~exist('../data/file_lists.mat', 'file')
        fprintf('\nERROR: Pipeline stopped at Step 1\n');
        return;
    end
    
    % Step 2: Extract 18 wavelet features
    step2_extract_features();
    if ~exist('../data/features_data.mat', 'file')
        fprintf('\nERROR: Pipeline stopped at Step 2\n');
        return;
    end
    
    % Step 3: Train SVM
    step3_train_svm();
    if ~exist('../models/svm_model.mat', 'file')
        fprintf('\nERROR: Pipeline stopped at Step 3\n');
        return;
    end
    
    % Step 4: Evaluate (TEST SET ONLY)
    step4_evaluate();
    
    % Step 5: Compare wavelets
    step5_compare_wavelets();
    
    % Step 6: Benchmark
    step6_benchmark();
    
    % Step 7: Compare lightweight classifier baselines
    step7_compare_baselines();
    
    % Plot results
    plot_results();
    
    elapsed_time = toc;
    
    fprintf('\n========================================================\n');
    fprintf('PIPELINE COMPLETED SUCCESSFULLY!\n');
    fprintf('Total execution time: %.2f seconds (%.2f minutes)\n', ...
        elapsed_time, elapsed_time/60);
    fprintf('========================================================\n');
    
end
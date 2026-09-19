%% step4_evaluate.m (v4 - no data leakage)
% Step 4: Detailed evaluation on TEST SET ONLY
% Compatible with MATLAB R2015b

function step4_evaluate()
    
    fprintf('\n========================================\n');
    fprintf('STEP 4: Evaluation (TEST SET ONLY)\n');
    fprintf('========================================\n\n');
    
    if ~exist('../results', 'dir')
        mkdir('../results');
    end
    
    if ~exist('../models/svm_model.mat', 'file')
        fprintf('ERROR: ../models/svm_model.mat not found!\n');
        return;
    end
    
    if ~exist('../data/features_data.mat', 'file')
        fprintf('ERROR: ../data/features_data.mat not found!\n');
        return;
    end
    
    load('../models/svm_model.mat', 'final_svm', 'mu', 'sigma', 'test_idx');
    load('../data/features_data.mat', 'all_features', 'all_labels');
    
    % ??? ONLY USE TEST SET (no data leakage) ???
    X_test = all_features(test_idx, :);
    Y_test = all_labels(test_idx);
    
    % Normalize using training mu/sigma
    X_test_norm = bsxfun(@minus, X_test, mu);
    X_test_norm = bsxfun(@rdivide, X_test_norm, sigma);
    X_test_norm(isnan(X_test_norm)) = 0;
    
    % Predict
    Y_pred = predict(final_svm, X_test_norm);
    [~, score] = predict(final_svm, X_test_norm);
    
    % Metrics
    TP = sum(Y_pred == 1 & Y_test == 1);
    TN = sum(Y_pred == 0 & Y_test == 0);
    FP = sum(Y_pred == 1 & Y_test == 0);
    FN = sum(Y_pred == 0 & Y_test == 1);
    
    accuracy = (TP + TN) / length(Y_test);
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1 = 2 * precision * recall / (precision + recall);
    
    [~, ~, ~, auc] = perfcurve(Y_test, score(:,2), 1);
    
    fprintf('========== TEST SET PERFORMANCE ==========\n');
    fprintf('Test samples: %d\n', length(Y_test));
    fprintf('Accuracy:  %.2f%%\n', accuracy * 100);
    fprintf('Precision: %.4f\n', precision);
    fprintf('Recall:    %.4f\n', recall);
    fprintf('F1:        %.4f\n', f1);
    fprintf('AUC-ROC:   %.4f\n', auc);
    
    % Save
    results.accuracy = accuracy;
    results.precision = precision;
    results.recall = recall;
    results.f1_score = f1;
    results.auc = auc;
    results.confusion_matrix = [TN, FP; FN, TP];
    results.test_size = length(Y_test);
    
    save('../results/evaluation_results.mat', 'results');
    fprintf('\nSaved ../results/evaluation_results.mat\n');
    
    % Feature importance (mean difference)
    fprintf('\n========== FEATURE ANALYSIS ==========\n');
    normal_features = all_features(all_labels == 0, :);
    anomaly_features = all_features(all_labels == 1, :);
    mean_diff = abs(mean(normal_features, 1) - mean(anomaly_features, 1));
    [sorted_diff, idx_sorted] = sort(mean_diff, 'descend');
    
    fprintf('Top 5 most discriminative features:\n');
    for i = 1:min(5, length(sorted_diff))
        fprintf('  Feature %d: mean difference = %.4f\n', ...
            idx_sorted(i), sorted_diff(i));
    end
    
end
%% step4_evaluate.m
% Step 4: Detailed evaluation and visualization
% Compatible with MATLAB R2015b

function step4_evaluate()
    
    fprintf('\n========================================\n');
    fprintf('STEP 4: Detailed Evaluation\n');
    fprintf('========================================\n\n');
    
    % Create results folder if not exists
    if ~exist('../results', 'dir')
        mkdir('../results');
    end
    
    % Load model and data
    if ~exist('../models/svm_model.mat', 'file')
        fprintf('ERROR: ../models/svm_model.mat not found!\n');
        fprintf('Please run step3_train_svm.m first.\n');
        return;
    end
    
    if ~exist('../data/features_data.mat', 'file')
        fprintf('ERROR: ../data/features_data.mat not found!\n');
        return;
    end
    
    load('../models/svm_model.mat', 'final_svm', 'mu', 'sigma');
    load('../data/features_data.mat', 'all_features', 'all_labels');
    
    % Normalize all data
    all_features_norm = bsxfun(@minus, all_features, mu);
    all_features_norm = bsxfun(@rdivide, all_features_norm, sigma);
    all_features_norm(isnan(all_features_norm)) = 0;
    
    % Get predictions for all samples
    all_pred = predict(final_svm, all_features_norm);
    
    % Calculate overall metrics
    TP_all = sum(all_pred == 1 & all_labels == 1);
    TN_all = sum(all_pred == 0 & all_labels == 0);
    FP_all = sum(all_pred == 1 & all_labels == 0);
    FN_all = sum(all_pred == 0 & all_labels == 1);
    
    accuracy_all = (TP_all + TN_all) / length(all_labels);
    precision_all = TP_all / (TP_all + FP_all);
    recall_all = TP_all / (TP_all + FN_all);
    f1_all = 2 * precision_all * recall_all / (precision_all + recall_all);
    
    % Calculate AUC-ROC
    [~, score] = predict(final_svm, all_features_norm);
    [~, ~, ~, auc] = perfcurve(all_labels, score(:,2), 1);
    
    fprintf('========== OVERALL PERFORMANCE ==========\n');
    fprintf('Total samples: %d\n', length(all_labels));
    fprintf('Accuracy:      %.2f%%\n', accuracy_all * 100);
    fprintf('Precision:     %.4f\n', precision_all);
    fprintf('Recall:        %.4f\n', recall_all);
    fprintf('F1-score:      %.4f\n', f1_all);
    fprintf('AUC-ROC:       %.4f\n', auc);
    
    % Feature importance analysis
    fprintf('\n========== FEATURE ANALYSIS ==========\n');
    
    % Separate features by class
    normal_features = all_features(all_labels == 0, :);
    anomaly_features = all_features(all_labels == 1, :);
    
    % Calculate mean difference for each feature
    mean_diff = abs(mean(normal_features, 1) - mean(anomaly_features, 1));
    [sorted_diff, idx_sorted] = sort(mean_diff, 'descend');
    
    fprintf('Top 5 most discriminative features:\n');
    for i = 1:min(5, length(sorted_diff))
        fprintf('  Feature %d: mean difference = %.4f\n', idx_sorted(i), sorted_diff(i));
    end
    
    % Save evaluation results
    results.accuracy = accuracy_all;
    results.precision = precision_all;
    results.recall = recall_all;
    results.f1_score = f1_all;
    results.auc = auc;
    results.confusion_matrix = [TN_all, FP_all; FN_all, TP_all];
    
    save('../results/evaluation_results.mat', 'results');
    fprintf('\nSaved ../results/evaluation_results.mat\n');
    
end
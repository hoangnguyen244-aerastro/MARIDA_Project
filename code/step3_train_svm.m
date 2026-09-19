%% step3_train_svm.m (v4 - balanced grid search)
% Step 3: Train SVM classifier with hyperparameter tuning
% Compatible with MATLAB R2015b

function step3_train_svm()
    
    fprintf('\n========================================\n');
    fprintf('STEP 3: Training SVM Classifier (v4)\n');
    fprintf('========================================\n\n');
    
    if ~exist('../models', 'dir')
        mkdir('../models');
    end
    
    if ~exist('../data/features_data.mat', 'file')
        fprintf('ERROR: ../data/features_data.mat not found!\n');
        return;
    end
    
    load('../data/features_data.mat', 'all_features', 'all_labels');
    
    n_samples = size(all_features, 1);
    n_features = size(all_features, 2);
    
    fprintf('Dataset: %d samples, %d features\n', n_samples, n_features);
    fprintf('Class distribution: Normal=%d, Anomaly=%d\n', ...
        sum(all_labels == 0), sum(all_labels == 1));
    
    if n_samples < 10
        fprintf('ERROR: Not enough data\n');
        return;
    end
    
    % ========== Split data ==========
    rng(42);
    indices = randperm(n_samples);
    train_size = floor(0.7 * n_samples);
    
    train_idx = indices(1:train_size);
    test_idx = indices(train_size+1:end);
    
    X_train = all_features(train_idx, :);
    Y_train = all_labels(train_idx);
    X_test = all_features(test_idx, :);
    Y_test = all_labels(test_idx);
    
    fprintf('\nSplit: %d training, %d test\n', length(Y_train), length(Y_test));
    
    % ========== Normalize features ==========
    fprintf('\nNormalizing features...\n');
    
    mu = mean(X_train, 1);
    sigma = std(X_train, 0, 1);
    sigma(sigma == 0) = 1;
    
    X_train_norm = bsxfun(@minus, X_train, mu);
    X_train_norm = bsxfun(@rdivide, X_train_norm, sigma);
    
    X_test_norm = bsxfun(@minus, X_test, mu);
    X_test_norm = bsxfun(@rdivide, X_test_norm, sigma);
    
    X_train_norm(isnan(X_train_norm)) = 0;
    X_test_norm(isnan(X_test_norm)) = 0;
    
    % ========== Grid search (balanced range) ==========
    fprintf('\nPerforming grid search (balanced range)...\n');
    
    % ??? BALANCED GRID SEARCH ???
    C_values = [0.1, 1, 10, 100];
    gamma_values = [0.01, 0.1, 1, 10];
    
    best_accuracy = 0;
    best_C = 1;
    best_gamma = 0.1;
    
    k_folds = min(5, length(Y_train));
    if k_folds < 3
        k_folds = 3;
    end
    
    for i = 1:length(C_values)
        for j = 1:length(gamma_values)
            try
                svm_temp = fitcsvm(X_train_norm, Y_train, ...
                    'KernelFunction', 'rbf', ...
                    'BoxConstraint', C_values(i), ...
                    'KernelScale', gamma_values(j), ...
                    'Standardize', false);
                
                cv_model = crossval(svm_temp, 'KFold', k_folds);
                cv_acc = 1 - kfoldLoss(cv_model);
                
                fprintf('  C=%.3f, gamma=%.4f -> CV accuracy = %.2f%%\n', ...
                    C_values(i), gamma_values(j), cv_acc * 100);
                
                if cv_acc > best_accuracy
                    best_accuracy = cv_acc;
                    best_C = C_values(i);
                    best_gamma = gamma_values(j);
                end
                
            catch ME
                fprintf('  C=%.3f, gamma=%.4f -> ERROR: %s\n', ...
                    C_values(i), gamma_values(j), ME.message);
            end
        end
    end
    
    fprintf('\nBest parameters: C = %.3f, gamma = %.4f\n', best_C, best_gamma);
    fprintf('Best CV accuracy: %.2f%%\n', best_accuracy * 100);
    
    % ========== Train final model ==========
    fprintf('\nTraining final SVM model...\n');
    
    final_svm = fitcsvm(X_train_norm, Y_train, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', best_C, ...
        'KernelScale', best_gamma, ...
        'Standardize', false);
    
    n_sv = size(final_svm.SupportVectors, 1);
    fprintf('Number of support vectors: %d (%.1f%% of training data)\n', ...
        n_sv, 100 * n_sv / length(Y_train));
    
    % ========== Evaluate on TEST set ==========
    Y_pred = predict(final_svm, X_test_norm);
    
    TP = sum(Y_pred == 1 & Y_test == 1);
    TN = sum(Y_pred == 0 & Y_test == 0);
    FP = sum(Y_pred == 1 & Y_test == 0);
    FN = sum(Y_pred == 0 & Y_test == 1);
    
    accuracy = (TP + TN) / length(Y_test);
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * (precision * recall) / (precision + recall);
    
    fprintf('\n========================================\n');
    fprintf('RESULTS (Test Set Only)\n');
    fprintf('========================================\n');
    fprintf('Test set size: %d images\n', length(Y_test));
    fprintf('Accuracy:      %.2f%%\n', accuracy * 100);
    fprintf('Precision:     %.4f\n', precision);
    fprintf('Recall:        %.4f\n', recall);
    fprintf('F1-score:      %.4f\n', f1_score);
    
    fprintf('\nConfusion Matrix:\n');
    fprintf('                Predicted\n');
    fprintf('                NORMAL  ANOMALY\n');
    fprintf('Actual NORMAL    %3d      %3d\n', TN, FP);
    fprintf('       ANOMALY    %3d      %3d\n', FN, TP);
    
    % ========== Save ==========
    save('../models/svm_model.mat', 'final_svm', 'mu', 'sigma', ...
         'best_C', 'best_gamma', 'train_idx', 'test_idx');
    fprintf('\nSaved ../models/svm_model.mat\n');
    
end
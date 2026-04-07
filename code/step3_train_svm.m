%% step3_train_svm.m
% Step 3: Train SVM classifier with hyperparameter tuning
% Compatible with MATLAB R2015b

function step3_train_svm()
    
    fprintf('\n========================================\n');
    fprintf('STEP 3: Training SVM Classifier\n');
    fprintf('========================================\n\n');
    
    % Create models folder if not exists
    if ~exist('../models', 'dir')
        mkdir('../models');
    end
    
    % Load features
    if ~exist('../data/features_data.mat', 'file')
        fprintf('ERROR: ../data/features_data.mat not found!\n');
        fprintf('Please run step2_extract_features.m first.\n');
        return;
    end
    
    load('../data/features_data.mat', 'all_features', 'all_labels');
    
    % Get dataset info
    n_samples = size(all_features, 1);
    n_features = size(all_features, 2);
    
    fprintf('Dataset: %d samples, %d features\n', n_samples, n_features);
    fprintf('Class distribution: Normal=%d, Anomaly=%d\n', ...
        sum(all_labels == 0), sum(all_labels == 1));
    
    % Check minimum data requirement
    if n_samples < 10
        fprintf('ERROR: Not enough data (need at least 10 samples)\n');
        return;
    end
    
    % ========== Split data into train (70%) and test (30%) ==========
    rng(42);  % For reproducible results
    indices = randperm(n_samples);
    train_size = floor(0.7 * n_samples);
    
    train_idx = indices(1:train_size);
    test_idx = indices(train_size+1:end);
    
    X_train = all_features(train_idx, :);
    Y_train = all_labels(train_idx);
    X_test = all_features(test_idx, :);
    Y_test = all_labels(test_idx);
    
    fprintf('\nSplit: %d training, %d test images\n', length(Y_train), length(Y_test));
    
    % ========== Normalize features (Z-score) ==========
    fprintf('\nNormalizing features...\n');
    
    % Calculate mean and std from training data
    mu = mean(X_train, 1);
    sigma = std(X_train, 0, 1);
    
    % Handle zero standard deviation
    sigma(sigma == 0) = 1;
    
    % Normalize using bsxfun (compatible with R2015b)
    X_train_norm = bsxfun(@minus, X_train, mu);
    X_train_norm = bsxfun(@rdivide, X_train_norm, sigma);
    
    X_test_norm = bsxfun(@minus, X_test, mu);
    X_test_norm = bsxfun(@rdivide, X_test_norm, sigma);
    
    % Handle any NaN values
    X_train_norm(isnan(X_train_norm)) = 0;
    X_test_norm(isnan(X_test_norm)) = 0;
    
    % ========== Hyperparameter grid search ==========
    fprintf('\nPerforming grid search for optimal parameters...\n');
    
    C_values = [0.1, 1, 10, 100];
    gamma_values = [0.01, 0.1, 1, 10];
    
    best_accuracy = 0;
    best_C = 1;
    best_gamma = 0.1;
    
    % Number of folds for cross-validation
    k_folds = min(5, length(Y_train));
    if k_folds < 3
        k_folds = 3;
    end
    
    for i = 1:length(C_values)
        for j = 1:length(gamma_values)
            try
                % Train SVM with current parameters
                svm_temp = fitcsvm(X_train_norm, Y_train, ...
                    'KernelFunction', 'rbf', ...
                    'BoxConstraint', C_values(i), ...
                    'KernelScale', gamma_values(j), ...
                    'Standardize', false);
                
                % Cross-validation
                cv_model = crossval(svm_temp, 'KFold', k_folds);
                cv_acc = 1 - kfoldLoss(cv_model);
                
                fprintf('  C=%.1f, gamma=%.2f -> CV accuracy = %.2f%%\n', ...
                    C_values(i), gamma_values(j), cv_acc * 100);
                
                % Update best parameters
                if cv_acc > best_accuracy
                    best_accuracy = cv_acc;
                    best_C = C_values(i);
                    best_gamma = gamma_values(j);
                end
                
            catch ME
                fprintf('  C=%.1f, gamma=%.2f -> ERROR: %s\n', ...
                    C_values(i), gamma_values(j), ME.message);
            end
        end
    end
    
    fprintf('\nBest parameters: C = %.1f, gamma = %.2f\n', best_C, best_gamma);
    fprintf('Best CV accuracy: %.2f%%\n', best_accuracy * 100);
    
    % ========== Train final model ==========
    fprintf('\nTraining final SVM model...\n');
    
    final_svm = fitcsvm(X_train_norm, Y_train, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', best_C, ...
        'KernelScale', best_gamma, ...
        'Standardize', false);
    
    % ========== Evaluate on test set ==========
    Y_pred = predict(final_svm, X_test_norm);
    
    % Calculate metrics
    TP = sum(Y_pred == 1 & Y_test == 1);
    TN = sum(Y_pred == 0 & Y_test == 0);
    FP = sum(Y_pred == 1 & Y_test == 0);
    FN = sum(Y_pred == 0 & Y_test == 1);
    
    accuracy = (TP + TN) / length(Y_test);
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * (precision * recall) / (precision + recall);
    
    % ========== Display results ==========
    fprintf('\n========================================\n');
    fprintf('RESULTS\n');
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
    
    % ========== Save model ==========
    save('../models/svm_model.mat', 'final_svm', 'mu', 'sigma', 'best_C', 'best_gamma');
    fprintf('\nSaved ../models/svm_model.mat\n');
    
end
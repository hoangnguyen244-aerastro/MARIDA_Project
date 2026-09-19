%% step5_compare_wavelets.m (FIXED - full dataset)
% Step 5: Compare different wavelet families on FULL dataset
% Compatible with MATLAB R2015b

function step5_compare_wavelets()
    
    fprintf('\n========================================\n');
    fprintf('STEP 5: Comparing Wavelet Families\n');
    fprintf('========================================\n\n');
    
    if ~exist('../results', 'dir')
        mkdir('../results');
    end
    
    if ~exist('../data/file_lists.mat', 'file')
        fprintf('ERROR: ../data/file_lists.mat not found!\n');
        return;
    end
    
    load('../data/file_lists.mat', 'normal_files', 'anomaly_files');
    
    wavelets = {'db4', 'sym4', 'bior3.5', 'coif2', 'haar'};
    num_wavelets = length(wavelets);
    
    % ??? CHANGED: Use FULL dataset (no 100-sample limit) ???
    num_normal = length(normal_files);
    num_anomaly = length(anomaly_files);
    
    fprintf('Using FULL dataset: %d normal, %d anomaly\n', ...
        num_normal, num_anomaly);
    
    results = struct();
    all_accuracies = zeros(1, num_wavelets);
    
    for w = 1:num_wavelets
        wavelet = wavelets{w};
        fprintf('\n>>> Testing wavelet: %s <<<\n', wavelet);
        
        all_features = [];
        all_labels = [];
        
        for i = 1:num_normal
            if mod(i, 50) == 0
                fprintf('  Normal: %d/%d\n', i, num_normal);
            end
            features = wavelet_feature_extractor(normal_files{i}, wavelet);
            features = sign(features) .* log1p(abs(features));
            all_features = [all_features; features];
            all_labels = [all_labels; 0];
        end
        
        for i = 1:num_anomaly
            if mod(i, 50) == 0
                fprintf('  Anomaly: %d/%d\n', i, num_anomaly);
            end
            features = wavelet_feature_extractor(anomaly_files{i}, wavelet);
            features = sign(features) .* log1p(abs(features));
            all_features = [all_features; features];
            all_labels = [all_labels; 1];
        end
        
        % Train/test split
        rng(42);
        n = size(all_features, 1);
        indices = randperm(n);
        train_idx = indices(1:floor(0.7*n));
        test_idx = indices(floor(0.7*n)+1:end);
        
        X_train = all_features(train_idx, :);
        Y_train = all_labels(train_idx);
        X_test = all_features(test_idx, :);
        Y_test = all_labels(test_idx);
        
        % Normalize
        mu = mean(X_train, 1);
        sigma = std(X_train, 0, 1);
        sigma(sigma == 0) = 1;
        
        X_train_norm = bsxfun(@minus, X_train, mu);
        X_train_norm = bsxfun(@rdivide, X_train_norm, sigma);
        X_test_norm = bsxfun(@minus, X_test, mu);
        X_test_norm = bsxfun(@rdivide, X_test_norm, sigma);
        
        X_train_norm(isnan(X_train_norm)) = 0;
        X_test_norm(isnan(X_test_norm)) = 0;
        
        % Train SVM
        svm = fitcsvm(X_train_norm, Y_train, 'KernelFunction', 'rbf');
        Y_pred = predict(svm, X_test_norm);
        
        acc = sum(Y_pred == Y_test) / length(Y_test);
        all_accuracies(w) = acc;
        
        results(w).wavelet = wavelet;
        results(w).accuracy = acc;
        
        fprintf('  Accuracy: %.2f%%\n', acc * 100);
    end
    
    % Display
    fprintf('\n========================================\n');
    fprintf('WAVELET COMPARISON RESULTS (FULL)\n');
    fprintf('========================================\n');
    fprintf('%-12s | %-10s\n', 'Wavelet', 'Accuracy');
    fprintf('-------------------------------\n');
    
    for w = 1:num_wavelets
        fprintf('%-12s | %-9.2f%%\n', ...
            results(w).wavelet, results(w).accuracy * 100);
    end
    
    [best_acc, best_idx] = max(all_accuracies);
    fprintf('\n>>> BEST WAVELET: %s (%.2f%%) <<<\n', ...
        results(best_idx).wavelet, best_acc * 100);
    
    % Export
    wavelet_names = cell(num_wavelets, 1);
    accuracy_values = zeros(num_wavelets, 1);
    for w = 1:num_wavelets
        wavelet_names{w} = results(w).wavelet;
        accuracy_values(w) = results(w).accuracy * 100;
    end
    
    excel_data = [wavelet_names, num2cell(accuracy_values)];
    excel_header = {'Wavelet', 'Accuracy_Percent'};
    excel_data_with_header = [excel_header; excel_data];
    
    try
        xlswrite('../results/wavelet_comparison.xls', excel_data_with_header);
        fprintf('\nSaved Excel file\n');
    catch
        fid = fopen('../results/wavelet_comparison.csv', 'w');
        fprintf(fid, '%s,%s\n', excel_header{:});
        for w = 1:num_wavelets
            fprintf(fid, '%s,%.2f\n', wavelet_names{w}, accuracy_values(w));
        end
        fclose(fid);
    end
    
    save('../results/wavelet_comparison.mat', 'results');
    fprintf('Saved ../results/wavelet_comparison.mat\n');
    
end
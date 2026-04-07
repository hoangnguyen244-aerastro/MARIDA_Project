%% step5_compare_wavelets.m
% Step 5: Compare different wavelet families
% Output: Excel file and MAT file with comparison results
% Compatible with MATLAB R2015b

function step5_compare_wavelets()
    
    fprintf('\n========================================\n');
    fprintf('STEP 5: Comparing Wavelet Families\n');
    fprintf('========================================\n\n');
    
    % Create results folder if not exists
    if ~exist('../results', 'dir')
        mkdir('../results');
    end
    
    % Load file lists
    if ~exist('../data/file_lists.mat', 'file')
        fprintf('ERROR: ../data/file_lists.mat not found!\n');
        fprintf('Please run step1_filter_data.m first.\n');
        return;
    end
    
    load('../data/file_lists.mat', 'normal_files', 'anomaly_files');
    
    % Wavelets to compare
    wavelets = {'db4', 'sym4', 'bior3.5', 'coif2', 'haar'};
    num_wavelets = length(wavelets);
    
    % Use subset for faster comparison
    max_samples = 100;
    num_normal = min(length(normal_files), max_samples);
    num_anomaly = min(length(anomaly_files), max_samples);
    
    normal_subset = normal_files(1:num_normal);
    anomaly_subset = anomaly_files(1:num_anomaly);
    
    fprintf('Using %d normal, %d anomaly images for comparison\n', ...
        num_normal, num_anomaly);
    
    % Store results
    results = struct();
    all_accuracies = zeros(1, num_wavelets);
    
    for w = 1:num_wavelets
        wavelet = wavelets{w};
        fprintf('\n>>> Testing wavelet: %s <<<\n', wavelet);
        
        % Extract features
        all_features = [];
        all_labels = [];
        
        % Normal images
        for i = 1:num_normal
            if mod(i, 20) == 0
                fprintf('  Normal: %d/%d\n', i, num_normal);
            end
            features = wavelet_feature_extractor(normal_subset{i}, wavelet);
            all_features = [all_features; features];
            all_labels = [all_labels; 0];
        end
        
        % Anomaly images
        for i = 1:num_anomaly
            if mod(i, 20) == 0
                fprintf('  Anomaly: %d/%d\n', i, num_anomaly);
            end
            features = wavelet_feature_extractor(anomaly_subset{i}, wavelet);
            all_features = [all_features; features];
            all_labels = [all_labels; 1];
        end
        
        % Simple train/test split
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
        
        % Train SVM
        svm = fitcsvm(X_train_norm, Y_train, 'KernelFunction', 'rbf');
        
        % Predict
        Y_pred = predict(svm, X_test_norm);
        
        % Calculate accuracy
        acc = sum(Y_pred == Y_test) / length(Y_test);
        all_accuracies(w) = acc;
        
        % Store
        results(w).wavelet = wavelet;
        results(w).accuracy = acc;
        
        fprintf('  Accuracy: %.2f%%\n', acc * 100);
    end
    
    % Display comparison table
    fprintf('\n========================================\n');
    fprintf('WAVELET COMPARISON RESULTS\n');
    fprintf('========================================\n');
    fprintf('%-12s | %-10s\n', 'Wavelet', 'Accuracy');
    fprintf('-------------------------------\n');
    
    for w = 1:num_wavelets
        fprintf('%-12s | %-9.2f%%\n', ...
            results(w).wavelet, results(w).accuracy * 100);
    end
    
    % Find best
    [best_acc, best_idx] = max(all_accuracies);
    fprintf('\n>>> BEST WAVELET: %s (%.2f%%) <<<\n', ...
        results(best_idx).wavelet, best_acc * 100);
    
    % ========== EXPORT TO EXCEL ==========
    % Create table for Excel export
    wavelet_names = cell(num_wavelets, 1);
    accuracy_values = zeros(num_wavelets, 1);
    
    for w = 1:num_wavelets
        wavelet_names{w} = results(w).wavelet;
        accuracy_values(w) = results(w).accuracy * 100;
    end
    
    % Prepare data for Excel
    excel_data = [wavelet_names, num2cell(accuracy_values)];
    excel_header = {'Wavelet', 'Accuracy_Percent'};
    excel_data_with_header = [excel_header; excel_data];
    
    % Save as Excel file
    excel_filename = '../results/wavelet_comparison.xls';
    try
        % Try to use xlswrite (available in R2015b)
        xlswrite(excel_filename, excel_data_with_header);
        fprintf('\nSaved Excel file: %s\n', excel_filename);
    catch
        % Fallback: save as CSV if Excel write fails
        csv_filename = '../results/wavelet_comparison.csv';
        fid = fopen(csv_filename, 'w');
        fprintf(fid, '%s,%s\n', excel_header{:});
        for w = 1:num_wavelets
            fprintf(fid, '%s,%.2f\n', wavelet_names{w}, accuracy_values(w));
        end
        fclose(fid);
        fprintf('\nSaved CSV file: %s (Excel fallback)\n', csv_filename);
    end
    
    % Also save as MAT file
    save('../results/wavelet_comparison.mat', 'results');
    fprintf('Saved ../results/wavelet_comparison.mat\n');
    
end
%% step2_extract_features.m (FIXED - with normalization)
% Step 2: Extract wavelet features from all images
% Features: Energy, Entropy, Skewness from 6 coefficient matrices (18 total)
% Compatible with MATLAB R2015b

function step2_extract_features()
    
    fprintf('\n========================================\n');
    fprintf('STEP 2: Extracting Wavelet Features\n');
    fprintf('========================================\n\n');
    
    % Create data folder if not exists
    if ~exist('../data', 'dir')
        mkdir('../data');
    end
    
    % Load file lists from Step 1
    if ~exist('../data/file_lists.mat', 'file')
        fprintf('ERROR: ../data/file_lists.mat not found!\n');
        fprintf('Please run step1_filter_data.m first.\n');
        return;
    end
    
    load('../data/file_lists.mat', 'normal_files', 'anomaly_files');
    
    % Get dataset sizes
    num_normal = length(normal_files);
    num_anomaly = length(anomaly_files);
    total_samples = num_normal + num_anomaly;
    
    fprintf('Normal images: %d\n', num_normal);
    fprintf('Anomaly images: %d\n', num_anomaly);
    fprintf('Total images to process: %d\n', total_samples);
    
    % Select wavelet type
    wavelet_name = 'db4';
    fprintf('\nUsing wavelet: %s\n', wavelet_name);
    
    % Initialize feature matrix and label vector
    all_features = zeros(total_samples, 18);
    all_labels = zeros(total_samples, 1);
    
    % ========== Process NORMAL images (label = 0) ==========
    fprintf('\n[1/2] Processing NORMAL images...\n');
    
    for i = 1:num_normal
        if mod(i, 20) == 0
            fprintf('  %d/%d\n', i, num_normal);
        end
        
        try
            features = wavelet_feature_extractor(normal_files{i}, wavelet_name);
            
            % FIX: Apply log transformation to handle large values
            features = sign(features) .* log1p(abs(features));
            
            % Check for NaN or Inf
            if any(isnan(features)) || any(isinf(features))
                fprintf('  WARNING: Invalid features for %s, using zeros\n', normal_files{i});
                features = zeros(1, 18);
            end
            
            all_features(i, :) = features;
            all_labels(i) = 0;
        catch ME
            fprintf('  ERROR: %s\n', ME.message);
            all_features(i, :) = zeros(1, 18);
            all_labels(i) = 0;
        end
    end
    
    % ========== Process ANOMALY images (label = 1) ==========
    fprintf('\n[2/2] Processing ANOMALY images...\n');
    
    for i = 1:num_anomaly
        if mod(i, 20) == 0
            fprintf('  %d/%d\n', i, num_anomaly);
        end
        
        try
            features = wavelet_feature_extractor(anomaly_files{i}, wavelet_name);
            
            % FIX: Apply log transformation to handle large values
            features = sign(features) .* log1p(abs(features));
            
            if any(isnan(features)) || any(isinf(features))
                fprintf('  WARNING: Invalid features for %s, using zeros\n', anomaly_files{i});
                features = zeros(1, 18);
            end
            
            all_features(num_normal + i, :) = features;
            all_labels(num_normal + i) = 1;
        catch ME
            fprintf('  ERROR: %s\n', ME.message);
            all_features(num_normal + i, :) = zeros(1, 18);
            all_labels(num_normal + i) = 1;
        end
    end
    
    % ========== Clean up ==========
    valid_rows = any(all_features ~= 0, 2);
    all_features = all_features(valid_rows, :);
    all_labels = all_labels(valid_rows);
    
    % Final normalization (Z-score across all features)
    for j = 1:size(all_features, 2)
        col_mean = mean(all_features(:, j));
        col_std = std(all_features(:, j));
        if col_std > 0
            all_features(:, j) = (all_features(:, j) - col_mean) / col_std;
        end
    end
    
    fprintf('\n========== EXTRACTION COMPLETE ==========\n');
    fprintf('Valid samples: %d\n', size(all_features, 1));
    fprintf('Feature dimension: %d\n', size(all_features, 2));
    fprintf('Class distribution: Normal=%d, Anomaly=%d\n', ...
        sum(all_labels == 0), sum(all_labels == 1));
    
    % Save features
    save('../data/features_data.mat', 'all_features', 'all_labels', 'wavelet_name');
    fprintf('\nSaved ../data/features_data.mat\n');
    
end
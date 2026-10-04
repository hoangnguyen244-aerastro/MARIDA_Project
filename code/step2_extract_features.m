%% step2_extract_features.m (v4 - 18 features)
% Step 2: Extract 18 wavelet features from all images
% Compatible with MATLAB R2015b

function step2_extract_features()
    
    fprintf('\n========================================\n');
    fprintf('STEP 2: Extracting Wavelet Features (18)\n');
    fprintf('========================================\n\n');
    
    if ~exist('../data', 'dir')
        mkdir('../data');
    end
    
    if ~exist('../data/file_lists.mat', 'file')
        fprintf('ERROR: ../data/file_lists.mat not found!\n');
        return;
    end
    
    load('../data/file_lists.mat', 'normal_files', 'anomaly_files');
    
    num_normal = length(normal_files);
    num_anomaly = length(anomaly_files);
    total_samples = num_normal + num_anomaly;
    
    fprintf('Normal images: %d\n', num_normal);
    fprintf('Anomaly images: %d\n', num_anomaly);
    fprintf('Total images to process: %d\n', total_samples);
    
    wavelet_name = 'db4';
    fprintf('\nUsing wavelet: %s\n', wavelet_name);
    
    % ??? 18 FEATURES ???
    NUM_FEATURES = 18;
    all_features = zeros(total_samples, NUM_FEATURES);
    all_labels = zeros(total_samples, 1);
    
    % ========== Process NORMAL images ==========
    fprintf('\n[1/2] Processing NORMAL images...\n');
    
    for i = 1:num_normal
        if mod(i, 50) == 0
            fprintf('  %d/%d\n', i, num_normal);
        end
        
        try
            features = wavelet_feature_extractor(normal_files{i}, wavelet_name);
            features = sign(features) .* log1p(abs(features));
            
            if any(isnan(features)) || any(isinf(features))
                features = zeros(1, NUM_FEATURES);
            end
            
            all_features(i, :) = features;
            all_labels(i) = 0;
        catch ME
            fprintf('  ERROR at %d: %s\n', i, ME.message);
            all_features(i, :) = zeros(1, NUM_FEATURES);
            all_labels(i) = 0;
        end
    end
    
    % ========== Process ANOMALY images ==========
    fprintf('\n[2/2] Processing ANOMALY images...\n');
    
    for i = 1:num_anomaly
        if mod(i, 50) == 0
            fprintf('  %d/%d\n', i, num_anomaly);
        end
        
        try
            features = wavelet_feature_extractor(anomaly_files{i}, wavelet_name);
            features = sign(features) .* log1p(abs(features));
            
            if any(isnan(features)) || any(isinf(features))
                features = zeros(1, NUM_FEATURES);
            end
            
            all_features(num_normal + i, :) = features;
            all_labels(num_normal + i) = 1;
        catch ME
            fprintf('  ERROR at %d: %s\n', i, ME.message);
            all_features(num_normal + i, :) = zeros(1, NUM_FEATURES);
            all_labels(num_normal + i) = 1;
        end
    end
    
    % Cleanup
    valid_rows = any(all_features ~= 0, 2);
    all_features = all_features(valid_rows, :);
    all_labels = all_labels(valid_rows);
    
    % IMPORTANT: do not normalize here. Normalization parameters must be
    % estimated from development/training data only after the official split
    % is applied (see step3_train_svm.m).
    
    fprintf('\n========== EXTRACTION COMPLETE ==========\n');
    fprintf('Valid samples: %d\n', size(all_features, 1));
    fprintf('Feature dimension: %d\n', size(all_features, 2));
    fprintf('Class distribution: Normal=%d, Anomaly=%d\n', ...
        sum(all_labels == 0), sum(all_labels == 1));
    
    save('../data/features_data.mat', 'all_features', 'all_labels', 'wavelet_name');
    fprintf('\nSaved ../data/features_data.mat\n');
    
end
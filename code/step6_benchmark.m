%% step6_benchmark.m
% Step 6: Benchmark model size, inference time per image, and save results figure
% Compatible with MATLAB R2015b

function step6_benchmark()
    
    fprintf('\n========================================\n');
    fprintf('STEP 6: Benchmarking Model Performance\n');
    fprintf('========================================\n\n');
    
    % Create reports folder if not exists
    if ~exist('../reports', 'dir')
        mkdir('../reports');
    end
    
    % ========== 1. MEASURE MODEL FILE SIZE ==========
    fprintf('[1/3] Measuring model file size...\n');
    
    model_file = '../models/svm_model.mat';
    if ~exist(model_file, 'file')
        fprintf('ERROR: Model file not found: %s\n', model_file);
        fprintf('Please run step3_train_svm.m first.\n');
        return;
    end
    
    file_info = dir(model_file);
    file_size_bytes = file_info.bytes;
    file_size_kb = file_size_bytes / 1024;
    file_size_mb = file_size_kb / 1024;
    
    fprintf('  Model file: %s\n', model_file);
    fprintf('  Size: %.2f KB (%.2f MB)\n', file_size_kb, file_size_mb);
    
    % ========== 2. LOAD MODEL AND DATA ==========
    fprintf('\n[2/3] Loading model and data...\n');
    
    load(model_file, 'final_svm', 'mu', 'sigma');
    load('../data/file_lists.mat', 'normal_files', 'anomaly_files');
    
    % Select a few images for benchmarking (10 normal + 10 anomaly)
    num_test = min(10, length(normal_files));
    test_files = [normal_files(1:num_test); anomaly_files(1:num_test)];
    num_images = length(test_files);
    
    fprintf('  Using %d images for timing (normal + anomaly)\n', num_images);
    
    % ========== 3. MEASURE INFERENCE TIME PER IMAGE ==========
    fprintf('\n[3/3] Measuring inference time (wavelet + predict)...\n');
    
    inference_times_ms = zeros(num_images, 1);
    wavelet_name = 'db4';  % same as used in training
    
    for i = 1:num_images
        img_path = test_files{i};
        
        % Start timing
        t_start = tic;
        
        % Step A: Extract wavelet features
        features = wavelet_feature_extractor(img_path, wavelet_name);
        
        % Step B: Normalize using training mu/sigma
        features_norm = (features - mu) ./ sigma;
        features_norm(isnan(features_norm)) = 0;
        
        % Step C: Predict using SVM
        label = predict(final_svm, features_norm);
        
        % End timing
        elapsed_sec = toc(t_start);
        elapsed_ms = elapsed_sec * 1000;
        inference_times_ms(i) = elapsed_ms;
        
        % Optional: display progress
        if mod(i, 5) == 0
            fprintf('  Processed %d/%d images\n', i, num_images);
        end
    end
    
    avg_time_ms = mean(inference_times_ms);
    std_time_ms = std(inference_times_ms);
    
    fprintf('\n========== INFERENCE TIME RESULTS ==========\n');
    fprintf('Average time per image: %.2f ms (%.4f seconds)\n', avg_time_ms, avg_time_ms/1000);
    fprintf('Standard deviation:     %.2f ms\n', std_time_ms);
    fprintf('Minimum time:           %.2f ms\n', min(inference_times_ms));
    fprintf('Maximum time:           %.2f ms\n', max(inference_times_ms));
    
    % ========== 4. SAVE RESULTS TO FIGURE ==========
    fprintf('\nSaving benchmark figure to reports/...\n');
    
    figure('Position', [100, 100, 700, 500]);
    
    % Subplot 1: Bar chart of inference times
    subplot(2,2,1);
    bar(inference_times_ms, 'FaceColor', [0.2 0.6 0.8]);
    xlabel('Image Index');
    ylabel('Time (ms)');
    title('Inference Time per Image');
    grid on;
    
    % Subplot 2: Boxplot of times
    subplot(2,2,2);
    boxplot(inference_times_ms);
    ylabel('Time (ms)');
    title('Distribution of Inference Times');
    grid on;
    
    % Subplot 3: Model size text
    subplot(2,2,3);
    axis off;
    text(0.1, 0.8, sprintf('Model Size: %.2f KB (%.2f MB)', file_size_kb, file_size_mb), ...
        'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.6, sprintf('Average Inference Time: %.2f ms', avg_time_ms), ...
        'FontSize', 12);
    text(0.1, 0.4, sprintf('Tested on %d images', num_images), ...
        'FontSize', 12);
    text(0.1, 0.2, sprintf('Wavelet: %s', wavelet_name), ...
        'FontSize', 12);
    
    % Subplot 4: Summary metrics
    subplot(2,2,4);
    axis off;
    text(0.1, 0.8, 'Performance Summary', 'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.6, sprintf('Min: %.2f ms', min(inference_times_ms)), 'FontSize', 11);
    text(0.1, 0.5, sprintf('Max: %.2f ms', max(inference_times_ms)), 'FontSize', 11);
    text(0.1, 0.4, sprintf('Std Dev: %.2f ms', std_time_ms), 'FontSize', 11);
    text(0.1, 0.2, sprintf('Frames per second: %.1f', 1000/avg_time_ms), 'FontSize', 11);
    
    % Save figure
    saveas(gcf, '../reports/benchmark_results.png');
    fprintf('  Saved: ../reports/benchmark_results.png\n');
    
    % Also save numeric results as .mat
    benchmark_results.model_size_kb = file_size_kb;
    benchmark_results.model_size_mb = file_size_mb;
    benchmark_results.avg_inference_ms = avg_time_ms;
    benchmark_results.std_inference_ms = std_time_ms;
    benchmark_results.min_ms = min(inference_times_ms);
    benchmark_results.max_ms = max(inference_times_ms);
    benchmark_results.num_test_images = num_images;
    benchmark_results.wavelet = wavelet_name;
    
    save('../results/benchmark_results.mat', 'benchmark_results');
    fprintf('  Saved: ../results/benchmark_results.mat\n');
    
    fprintf('\n========== BENCHMARK COMPLETED ==========\n');
    
end
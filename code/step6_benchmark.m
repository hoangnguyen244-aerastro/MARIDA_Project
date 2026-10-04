%% step6_benchmark.m
% Step 6: Benchmark model size, inference time per image
% Compatible with MATLAB R2015b

function step6_benchmark()
    
    fprintf('\n========================================\n');
    fprintf('STEP 6: Benchmarking Model Performance\n');
    fprintf('========================================\n\n');
    
    if ~exist('../reports', 'dir')
        mkdir('../reports');
    end
    
    % ========== 1. MEASURE MODEL FILE SIZE ==========
    fprintf('[1/3] Measuring model file size...\n');
    
    model_file = '../models/svm_model.mat';
    if ~exist(model_file, 'file')
        fprintf('ERROR: Model file not found: %s\n', model_file);
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
    
    load(model_file, 'final_svm', 'mu', 'sigma', 'test_idx');
    load('../data/features_data.mat', 'valid_files');
    test_files = valid_files(test_idx);
    num_images = length(test_files);
    
    fprintf('  Using %d images for timing\n', num_images);
    
    % ========== 3. MEASURE INFERENCE TIME ==========
    fprintf('\n[3/3] Measuring inference time...\n');
    
    inference_times_ms = zeros(num_images, 1);
    wavelet_name = 'db4'; % benchmark the deployed main model; keep synchronized with step3
    
    for i = 1:num_images
        img_path = test_files{i};
        
        t_start = tic;
        features = wavelet_feature_extractor(img_path, wavelet_name);
        features = sign(features) .* log1p(abs(features));
        features_norm = (features - mu) ./ sigma;
        features_norm(~isfinite(features_norm)) = 0;
        label = predict(final_svm, features_norm);
        elapsed_sec = toc(t_start);
        
        inference_times_ms(i) = elapsed_sec * 1000;
        
        if mod(i, 5) == 0
            fprintf('  Processed %d/%d images\n', i, num_images);
        end
    end
    
    avg_time_ms = mean(inference_times_ms);
    std_time_ms = std(inference_times_ms);
    
    fprintf('\n========== INFERENCE TIME RESULTS ==========\n');
    fprintf('Average time per image: %.2f ms\n', avg_time_ms);
    fprintf('Standard deviation:     %.2f ms\n', std_time_ms);
    fprintf('Minimum time:           %.2f ms\n', min(inference_times_ms));
    fprintf('Maximum time:           %.2f ms\n', max(inference_times_ms));
    fprintf('Frames per second:      %.1f\n', 1000/avg_time_ms);
    
    % ========== 4. SAVE RESULTS ==========
    benchmark_results.model_size_kb = file_size_kb;
    benchmark_results.model_size_mb = file_size_mb;
    benchmark_results.avg_inference_ms = avg_time_ms;
    benchmark_results.std_inference_ms = std_time_ms;
    benchmark_results.min_ms = min(inference_times_ms);
    benchmark_results.max_ms = max(inference_times_ms);
    benchmark_results.num_test_images = num_images;
    benchmark_results.wavelet = wavelet_name;
    benchmark_results.fps = 1000/avg_time_ms;
    
    save('../results/benchmark_results.mat', 'benchmark_results');
    fprintf('\nSaved: ../results/benchmark_results.mat\n');
    
    % Figure
    figure('Position', [100, 100, 700, 500]);
    
    subplot(2,2,1);
    bar(inference_times_ms, 'FaceColor', [0.2 0.6 0.8]);
    xlabel('Image Index'); ylabel('Time (ms)');
    title('Inference Time per Image'); grid on;
    
    subplot(2,2,2);
    boxplot(inference_times_ms);
    ylabel('Time (ms)');
    title('Distribution of Inference Times'); grid on;
    
    subplot(2,2,3);
    axis off;
    text(0.1, 0.8, sprintf('Model Size: %.2f KB', file_size_kb), ...
        'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.6, sprintf('Avg Inference: %.2f ms', avg_time_ms), 'FontSize', 12);
    text(0.1, 0.4, sprintf('Tested on %d images', num_images), 'FontSize', 12);
    text(0.1, 0.2, sprintf('Wavelet: %s', wavelet_name), 'FontSize', 12);
    
    subplot(2,2,4);
    axis off;
    text(0.1, 0.8, 'Performance Summary', 'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.6, sprintf('Min: %.2f ms', min(inference_times_ms)), 'FontSize', 11);
    text(0.1, 0.5, sprintf('Max: %.2f ms', max(inference_times_ms)), 'FontSize', 11);
    text(0.1, 0.4, sprintf('Std Dev: %.2f ms', std_time_ms), 'FontSize', 11);
    text(0.1, 0.2, sprintf('FPS: %.1f', 1000/avg_time_ms), 'FontSize', 11);
    
    saveas(gcf, '../reports/benchmark_results.png');
    fprintf('  Saved: ../reports/benchmark_results.png\n');
    
    fprintf('\n========== BENCHMARK COMPLETED ==========\n');
    
end
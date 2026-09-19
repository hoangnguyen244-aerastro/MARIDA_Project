%% plot_results.m
% Generate all figures for the paper
% Compatible with MATLAB R2015b

function plot_results()
    
    fprintf('\n========================================\n');
    fprintf('GENERATING FIGURES FOR PAPER\n');
    fprintf('========================================\n\n');
    
    if ~exist('../reports', 'dir')
        mkdir('../reports');
    end
    
    % ========== FIGURE 1: Wavelet Comparison ==========
    fprintf('Generating Figure 1: Wavelet Comparison...\n');
    
    if exist('../results/wavelet_comparison.mat', 'file')
        load('../results/wavelet_comparison.mat', 'results');
        
        num_wavelets = length(results);
        wavelet_names = cell(1, num_wavelets);
        accuracies = zeros(1, num_wavelets);
        
        for i = 1:num_wavelets
            wavelet_names{i} = results(i).wavelet;
            accuracies(i) = results(i).accuracy * 100;
        end
        
        figure('Position', [100, 100, 600, 450]);
        bar(accuracies, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'k');
        
        set(gca, 'XTickLabel', wavelet_names, 'FontSize', 11);
        xlabel('Wavelet Family', 'FontSize', 12);
        ylabel('Accuracy (%)', 'FontSize', 12);
        title('Wavelet Comparison for Anomaly Detection', 'FontSize', 14);
        ylim([0, 100]);
        grid on;
        
        for i = 1:num_wavelets
            text(i, accuracies(i) + 1.5, sprintf('%.1f%%', accuracies(i)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
        end
        
        saveas(gcf, '../reports/wavelet_comparison.png');
        fprintf('  Saved: ../reports/wavelet_comparison.png\n');
        close(gcf);
    end
    
    % ========== FIGURE 2: Confusion Matrix ==========
    fprintf('Generating Figure 2: Confusion Matrix...\n');
    
    if exist('../results/evaluation_results.mat', 'file')
        load('../results/evaluation_results.mat', 'results');
        CM = results.confusion_matrix;
        
        figure('Position', [100, 100, 450, 400]);
        imagesc(CM);
        colormap(flipud(gray));
        colorbar;
        
        for i = 1:2
            for j = 1:2
                text(j, i, sprintf('%d', CM(i,j)), ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'FontSize', 16, 'FontWeight', 'bold', 'Color', 'r');
            end
        end
        
        set(gca, 'XTick', [1, 2], 'XTickLabel', {'Normal', 'Anomaly'});
        set(gca, 'YTick', [1, 2], 'YTickLabel', {'Normal', 'Anomaly'});
        xlabel('Predicted Class', 'FontSize', 12);
        ylabel('Actual Class', 'FontSize', 12);
        title(sprintf('Confusion Matrix (Accuracy: %.1f%%)', results.accuracy * 100), ...
            'FontSize', 14);
        
        saveas(gcf, '../reports/confusion_matrix.png');
        fprintf('  Saved: ../reports/confusion_matrix.png\n');
        close(gcf);
    end
    
    % ========== FIGURE 3: Feature Importance ==========
    fprintf('Generating Figure 3: Feature Importance...\n');
    
    if exist('../data/features_data.mat', 'file')
        load('../data/features_data.mat', 'all_features', 'all_labels');
        
        normal_mean = mean(all_features(all_labels == 0, :), 1);
        anomaly_mean = mean(all_features(all_labels == 1, :), 1);
        mean_diff = abs(normal_mean - anomaly_mean);
        mean_diff_pct = 100 * mean_diff / max(mean_diff);
        
        [sorted_diff, idx] = sort(mean_diff_pct, 'descend');
        top10_idx = idx(1:min(10, length(idx)));
        top10_values = sorted_diff(1:min(10, length(idx)));
        
        figure('Position', [100, 100, 600, 400]);
        bar(top10_values, 'FaceColor', [0.8, 0.4, 0.2], 'EdgeColor', 'k');
        
        xlabel('Feature Index', 'FontSize', 12);
        ylabel('Discriminative Power (%)', 'FontSize', 12);
        title('Top 10 Most Discriminative Features', 'FontSize', 14);
        set(gca, 'XTick', 1:length(top10_idx));
        set(gca, 'XTickLabel', cellstr(num2str(top10_idx')));
        grid on;
        ylim([0, 105]);
        
        saveas(gcf, '../reports/feature_importance.png');
        fprintf('  Saved: ../reports/feature_importance.png\n');
        close(gcf);
    end
    
    % ========== FIGURE 4: ROC Curve ==========
    fprintf('Generating Figure 4: ROC Curve...\n');
    
    if exist('../models/svm_model.mat', 'file') && exist('../data/features_data.mat', 'file')
        load('../models/svm_model.mat', 'final_svm', 'mu', 'sigma', 'test_idx');
        load('../data/features_data.mat', 'all_features', 'all_labels');
        
        % ? CH? DÙNG TEST SET ?
        X_test = all_features(test_idx, :);
        Y_test = all_labels(test_idx);
        
        X_test_norm = bsxfun(@minus, X_test, mu);
        X_test_norm = bsxfun(@rdivide, X_test_norm, sigma);
        X_test_norm(isnan(X_test_norm)) = 0;
        
        [~, score] = predict(final_svm, X_test_norm);
        
        [X_Y, Y_X, ~, AUC] = perfcurve(Y_test, score(:,2), 1);
        
        figure('Position', [100, 100, 500, 450]);
        plot(X_Y, Y_X, 'b-', 'LineWidth', 2);
        hold on;
        plot([0, 1], [0, 1], 'r--', 'LineWidth', 1);
        xlabel('False Positive Rate', 'FontSize', 12);
        ylabel('True Positive Rate', 'FontSize', 12);
        title(sprintf('ROC Curve (AUC = %.3f)', AUC), 'FontSize', 14);
        legend({'SVM', 'Random Classifier'}, 'Location', 'southeast');
        grid on;
        xlim([0, 1]); ylim([0, 1]);
        
        saveas(gcf, '../reports/roc_curve.png');
        fprintf('  Saved: ../reports/roc_curve.png\n');
        close(gcf);
    end
    
    fprintf('\n========================================\n');
    fprintf('ALL FIGURES GENERATED!\n');
    fprintf('========================================\n');
    
end
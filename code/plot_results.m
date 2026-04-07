%% plot_results.m
% Script to generate all figures for the paper
% Reads results from step5 and creates plots in reports folder
% Compatible with MATLAB R2015b
% Run this AFTER step5_compare_wavelets.m

function plot_results()
    
    fprintf('\n========================================\n');
    fprintf('GENERATING FIGURES FOR PAPER\n');
    fprintf('========================================\n\n');
    
    % Create reports folder if not exists
    if ~exist('../reports', 'dir')
        mkdir('../reports');
    end
    
    % ========== FIGURE 1: Wavelet Comparison Bar Chart ==========
    fprintf('Generating Figure 1: Wavelet Comparison...\n');
    
    % Load wavelet comparison results
    if exist('../results/wavelet_comparison.mat', 'file')
        load('../results/wavelet_comparison.mat', 'results');
        
        % Extract data
        num_wavelets = length(results);
        wavelet_names = cell(1, num_wavelets);
        accuracies = zeros(1, num_wavelets);
        
        for i = 1:num_wavelets
            wavelet_names{i} = results(i).wavelet;
            accuracies(i) = results(i).accuracy * 100;
        end
        
        % Create bar chart
        figure('Position', [100, 100, 600, 450]);
        bar(accuracies, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'k');
        
        % Customize
        set(gca, 'XTickLabel', wavelet_names, 'FontSize', 11);
        xlabel('Wavelet Family', 'FontSize', 12);
        ylabel('Accuracy (%)', 'FontSize', 12);
        title('Wavelet Comparison for Anomaly Detection', 'FontSize', 14);
        ylim([0, 100]);
        grid on;
        
        % Add value labels on top of bars
        for i = 1:num_wavelets
            text(i, accuracies(i) + 1.5, sprintf('%.1f%%', accuracies(i)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
        end
        
        % Save
        saveas(gcf, '../reports/wavelet_comparison.png');
        fprintf('  Saved: ../reports/wavelet_comparison.png\n');
        close(gcf);
    else
        fprintf('  WARNING: wavelet_comparison.mat not found. Run step5 first.\n');
    end
    
    % ========== FIGURE 2: Confusion Matrix Heatmap ==========
    fprintf('Generating Figure 2: Confusion Matrix...\n');
    
    if exist('../results/evaluation_results.mat', 'file')
        load('../results/evaluation_results.mat', 'results');
        
        % Get confusion matrix
        CM = results.confusion_matrix;
        
        % Create heatmap-style figure
        figure('Position', [100, 100, 450, 400]);
        
        % Display as image
        imagesc(CM);
        colormap(flipud(gray));
        colorbar;
        
        % Add text labels
        for i = 1:2
            for j = 1:2
                text(j, i, sprintf('%d', CM(i,j)), ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'FontSize', 16, 'FontWeight', 'bold', ...
                    'Color', 'r');
            end
        end
        
        % Customize
        set(gca, 'XTick', [1, 2], 'XTickLabel', {'Normal', 'Anomaly'});
        set(gca, 'YTick', [1, 2], 'YTickLabel', {'Normal', 'Anomaly'});
        xlabel('Predicted Class', 'FontSize', 12);
        ylabel('Actual Class', 'FontSize', 12);
        title(sprintf('Confusion Matrix (Accuracy: %.1f%%)', results.accuracy * 100), ...
            'FontSize', 14);
        
        % Save
        saveas(gcf, '../reports/confusion_matrix.png');
        fprintf('  Saved: ../reports/confusion_matrix.png\n');
        close(gcf);
    else
        fprintf('  WARNING: evaluation_results.mat not found. Run step4 first.\n');
    end
    
    % ========== FIGURE 3: Feature Importance ==========
    fprintf('Generating Figure 3: Feature Importance...\n');
    
    if exist('../data/features_data.mat', 'file')
        load('../data/features_data.mat', 'all_features', 'all_labels');
        
        % Calculate mean difference per feature
        normal_mean = mean(all_features(all_labels == 0, :), 1);
        anomaly_mean = mean(all_features(all_labels == 1, :), 1);
        mean_diff = abs(normal_mean - anomaly_mean);
        
        % Normalize to percentage
        mean_diff_pct = 100 * mean_diff / max(mean_diff);
        
        % Take top 10 features
        [sorted_diff, idx] = sort(mean_diff_pct, 'descend');
        top10_idx = idx(1:min(10, length(idx)));
        top10_values = sorted_diff(1:min(10, length(idx)));
        
        % Create bar chart
        figure('Position', [100, 100, 600, 400]);
        bar(top10_values, 'FaceColor', [0.8, 0.4, 0.2], 'EdgeColor', 'k');
        
        % Customize
        xlabel('Feature Index', 'FontSize', 12);
        ylabel('Discriminative Power (%)', 'FontSize', 12);
        title('Top 10 Most Discriminative Features', 'FontSize', 14);
        set(gca, 'XTick', 1:length(top10_idx));
        set(gca, 'XTickLabel', cellstr(num2str(top10_idx')));
        grid on;
        ylim([0, 105]);
        
        % Save
        saveas(gcf, '../reports/feature_importance.png');
        fprintf('  Saved: ../reports/feature_importance.png\n');
        close(gcf);
    else
        fprintf('  WARNING: features_data.mat not found.\n');
    end
    
    % ========== FIGURE 4: ROC Curve ==========
    fprintf('Generating Figure 4: ROC Curve...\n');
    
    if exist('../models/svm_model.mat', 'file') && exist('../data/features_data.mat', 'file')
        load('../models/svm_model.mat', 'final_svm', 'mu', 'sigma');
        load('../data/features_data.mat', 'all_features', 'all_labels');
        
        % Normalize data
        all_features_norm = bsxfun(@minus, all_features, mu);
        all_features_norm = bsxfun(@rdivide, all_features_norm, sigma);
        all_features_norm(isnan(all_features_norm)) = 0;
        
        % Get scores
        [~, score] = predict(final_svm, all_features_norm);
        
        % Compute ROC
        [X_Y, Y_X, T, AUC] = perfcurve(all_labels, score(:,2), 1);
        
        % Plot ROC
        figure('Position', [100, 100, 500, 450]);
        plot(X_Y, Y_X, 'b-', 'LineWidth', 2);
        hold on;
        plot([0, 1], [0, 1], 'r--', 'LineWidth', 1);
        xlabel('False Positive Rate (1 - Specificity)', 'FontSize', 12);
        ylabel('True Positive Rate (Sensitivity)', 'FontSize', 12);
        title(sprintf('ROC Curve (AUC = %.3f)', AUC), 'FontSize', 14);
        legend({'SVM', 'Random Classifier'}, 'Location', 'southeast');
        grid on;
        xlim([0, 1]);
        ylim([0, 1]);
        
        % Save
        saveas(gcf, '../reports/roc_curve.png');
        fprintf('  Saved: ../reports/roc_curve.png\n');
        close(gcf);
    else
        fprintf('  WARNING: Cannot generate ROC curve.\n');
    end
    
    fprintf('\n========================================\n');
    fprintf('ALL FIGURES GENERATED SUCCESSFULLY!\n');
    fprintf('Figures saved to: ../reports/\n');
    fprintf('========================================\n');
    
end
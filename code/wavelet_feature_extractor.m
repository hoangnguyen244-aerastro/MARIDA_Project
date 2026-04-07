%% wavelet_feature_extractor.m
% Extract wavelet-based statistical features from an image
% Compatible with MATLAB R2015b

function features = wavelet_feature_extractor(img_path, wavelet_name)
    % Read image
    I = imread(img_path);
    
    % Convert to grayscale if RGB
    if ndims(I) == 3
        % Standard RGB to grayscale conversion
        I = 0.2989 * I(:,:,1) + 0.5870 * I(:,:,2) + 0.1140 * I(:,:,3);
    end
    
    % Convert to double precision
    I = im2double(I);
    
    % 2-level Discrete Wavelet Transform
    [C, S] = wavedec2(I, 2, wavelet_name);
    
    % Extract detail coefficients at level 1 and 2
    [H1, V1, D1] = detcoef2('all', C, S, 1);
    [H2, V2, D2] = detcoef2('all', C, S, 2);
    
    % Collect all coefficient matrices
    coeff_cells = {H1, V1, D1, H2, V2, D2};
    
    % Initialize feature vector
    features = zeros(1, 18);
    feat_idx = 1;
    
    % Process each coefficient matrix
    for k = 1:length(coeff_cells)
        coeff = coeff_cells{k};
        coeff_vec = coeff(:);
        
        % Feature 1: Energy
        energy = sum(coeff_vec .^ 2);
        features(feat_idx) = energy;
        
        % Feature 2: Entropy
        p = abs(coeff_vec) / sum(abs(coeff_vec));
        p(p == 0) = [];
        if isempty(p)
            entropy_val = 0;
        else
            entropy_val = -sum(p .* log2(p));
        end
        features(feat_idx + 1) = entropy_val;
        
        % Feature 3: Skewness
        n = length(coeff_vec);
        if n < 3
            skewness_val = 0;
        else
            mean_coeff = mean(coeff_vec);
            std_coeff = std(coeff_vec);
            if std_coeff == 0
                skewness_val = 0;
            else
                skewness_val = (sum((coeff_vec - mean_coeff).^3) / n) / (std_coeff^3);
            end
        end
        features(feat_idx + 2) = skewness_val;
        
        feat_idx = feat_idx + 3;
    end
end
%% wavelet_feature_extractor.m (FIXED - 30 features)
% Extract wavelet-based statistical features from an image
% Features per coefficient matrix (5): Energy, Entropy, Skewness, Kurtosis, Spectral Slope
% 6 coefficient matrices × 5 features = 30 features total
% Compatible with MATLAB R2015b

function features = wavelet_feature_extractor(img_path, wavelet_name)
    % Read image
    I = imread(img_path);
    
    if ndims(I) == 3
        I = 0.2989 * I(:,:,1) + 0.5870 * I(:,:,2) + 0.1140 * I(:,:,3);
    end
    
    I = im2double(I);
    
    % 2-level Discrete Wavelet Transform
    [C, S] = wavedec2(I, 2, wavelet_name);
    
    % Extract detail coefficients at level 1 and 2
    [H1, V1, D1] = detcoef2('all', C, S, 1);
    [H2, V2, D2] = detcoef2('all', C, S, 2);
    
    % Also include approximation at level 2 (LL2) for global context
    LL2 = appcoef2(C, S, wavelet_name, 2);
    
    % ? 6 coefficient matrices (same as before, but now LL2 replaces nothing)
    % Actually keep original 6: H1, V1, D1, H2, V2, D2
    coeff_cells = {H1, V1, D1, H2, V2, D2};
    
    % ??? CHANGED: 18 ? 30 features (6 matrices × 5 features) ???
    features = zeros(1, 30);
    feat_idx = 1;
    
    for k = 1:length(coeff_cells)
        coeff = coeff_cells{k};
        coeff_vec = coeff(:);
        n = length(coeff_vec);
        
        % ---- Feature 1: Energy ----
        energy = sum(coeff_vec .^ 2);
        features(feat_idx) = energy;
        
        % ---- Feature 2: Entropy (normalized energy entropy) ----
        p = abs(coeff_vec) / sum(abs(coeff_vec));
        p(p == 0) = [];
        if isempty(p)
            entropy_val = 0;
        else
            entropy_val = -sum(p .* log2(p));
        end
        features(feat_idx + 1) = entropy_val;
        
        % ---- Feature 3: Skewness ----
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
        
        % ---- Feature 4: ? NEW - Kurtosis ? ----
        if n < 4
            kurtosis_val = 0;
        else
            mean_coeff = mean(coeff_vec);
            std_coeff = std(coeff_vec);
            if std_coeff == 0
                kurtosis_val = 0;
            else
                % Excess kurtosis (Fisher's definition, normal = 0)
                kurtosis_val = (sum((coeff_vec - mean_coeff).^4) / n) / (std_coeff^4) - 3;
            end
        end
        features(feat_idx + 3) = kurtosis_val;
        
        % ---- Feature 5: ? NEW - Spectral Slope ? ----
        % Compute radially-averaged power spectrum and fit log-log slope
        spectral_slope = compute_spectral_slope(coeff);
        features(feat_idx + 4) = spectral_slope;
        
        feat_idx = feat_idx + 5;
    end
end

% ========== Helper function: Spectral Slope ==========
function slope = compute_spectral_slope(coeff)
    % Compute 2D FFT power spectrum
    [M, N] = size(coeff);
    if M < 4 || N < 4
        slope = 0;
        return;
    end
    
    F = fft2(coeff);
    P = abs(fftshift(F)).^2;
    
    % Compute radial frequency
    [fy, fx] = meshgrid(-floor(N/2):ceil(N/2)-1, -floor(M/2):ceil(M/2)-1);
    fr = sqrt(fx.^2 + fy.^2);
    
    % Bin radial frequencies
    max_r = min(floor(M/2), floor(N/2));
    if max_r < 2
        slope = 0;
        return;
    end
    
    r_bins = 1:max_r;
    P_radial = zeros(size(r_bins));
    for r = r_bins
        mask = (fr >= r-0.5) & (fr < r+0.5);
        if any(mask(:))
            P_radial(r) = mean(P(mask));
        end
    end
    
    % Remove zero/NaN values
    valid = P_radial > 0 & ~isnan(P_radial);
    if sum(valid) < 3
        slope = 0;
        return;
    end
    
    % Log-log linear fit
    log_r = log(r_bins(valid));
    log_P = log(P_radial(valid));
    
    p = polyfit(log_r, log_P, 1);
    slope = p(1);
    
    % Clip extreme values (robustness)
    if isnan(slope) || isinf(slope)
        slope = 0;
    end
    slope = max(min(slope, 10), -10);
end
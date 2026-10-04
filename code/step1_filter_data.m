%% step1_filter_data.m (v5 - FULL MARIDA, NATURAL CLASS DISTRIBUTION)
% Step 1: Filter MARIDA dataset without global class balancing.
% Normal: any image containing water (class 7)
% Anomaly: any image containing debris (class 1) OR ship (class 5)
% Compatible with MATLAB R2015b

function step1_filter_data()
    
    fprintf('\n========================================\n');
    fprintf('STEP 1: Filtering MARIDA (FULL)\n');
    fprintf('========================================\n\n');
    
    % ==================== CONFIGURATION ====================
    root_path = '../raw_data/MARIDA/';
    % =======================================================
    
    patches_path = fullfile(root_path, 'patches');
    
    if ~exist(patches_path, 'dir')
        fprintf('ERROR: Path does not exist: %s\n', patches_path);
        return;
    end
    
    scene_dirs = dir(patches_path);
    scene_dirs = scene_dirs([scene_dirs.isdir]);
    scene_dirs = scene_dirs(~ismember({scene_dirs.name}, {'.', '..'}));
    
    fprintf('Found %d scene directories\n', length(scene_dirs));
    
    normal_files = {};
    anomaly_files = {};
    
    total_checked = 0;
    num_normal = 0;
    num_anomaly = 0;
    
    fprintf('\nScanning images (FULL DATASET - no limit)...\n');
    
    for s = 1:length(scene_dirs)
        scene_path = fullfile(patches_path, scene_dirs(s).name);
        mask_files = dir(fullfile(scene_path, '*_cl.tif'));
        
        for f = 1:length(mask_files)
            mask_filename = mask_files(f).name;
            total_checked = total_checked + 1;
            
            basename = strrep(mask_filename, '_cl.tif', '');
            img_filename = [basename '.tif'];
            img_path = fullfile(scene_path, img_filename);
            
            if ~exist(img_path, 'file')
                continue;
            end
            
            mask = imread(fullfile(scene_path, mask_filename));
            
            has_debris = any(mask(:) == 1);
            has_ship = any(mask(:) == 5);
            is_anomaly = has_debris || has_ship;
            has_water = any(mask(:) == 7);
            
            if is_anomaly
                anomaly_files{end+1} = img_path;
                num_anomaly = num_anomaly + 1;
            elseif has_water
                normal_files{end+1} = img_path;
                num_normal = num_normal + 1;
            end
            
            if mod(total_checked, 100) == 0
                fprintf('  Processed %d masks... (Normal: %d, Anomaly: %d)\n', ...
                    total_checked, num_normal, num_anomaly);
            end
        end
    end
    
    fprintf('\n========== SCAN COMPLETE ==========\n');
    fprintf('Total masks checked: %d\n', total_checked);
    fprintf('NORMAL images found: %d\n', num_normal);
    fprintf('ANOMALY images found: %d\n', num_anomaly);
    
    if num_normal == 0 || num_anomaly == 0
        fprintf('\nERROR: One class is empty!\n');
        return;
    end
    
    % Preserve the natural eligible class distribution.
    % Imbalance handling, if ever needed, belongs only inside development/training.
    % The official test set must never be undersampled based on its labels.
    fprintf('\nKeeping natural distribution: %d NORMAL, %d ANOMALY (total = %d)\n', ...
        length(normal_files), length(anomaly_files), length(normal_files)+length(anomaly_files));

    if ~exist('../data', 'dir')
        mkdir('../data');
    end
    
    save('../data/file_lists.mat', 'normal_files', 'anomaly_files');
    fprintf('\nSaved ../data/file_lists.mat\n');
    
    fprintf('\nSample NORMAL: %s\n', normal_files{1});
    fprintf('Sample ANOMALY: %s\n', anomaly_files{1});
    
end
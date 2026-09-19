%% step1_filter_data.m
% Step 1: Filter and balance MARIDA dataset
% Filter: 
%   - Normal: any image containing water (class 7)
%   - Anomaly: any image containing debris (class 1) OR ship (class 5)
% Compatible with MATLAB R2015b

function step1_filter_data()
    
    fprintf('\n========================================\n');
    fprintf('STEP 1: Filtering MARIDA Dataset\n');
    fprintf('========================================\n\n');
    
    % ==================== CONFIGURATION ====================
    % Using relative path - raw_data folder inside project
    %root_path = 'E:\SpbSUITD\Semester2\Neural-Network\MARIDA_Project\raw_data\MARIDA\';
    root_path = '../raw_data/MARIDA/';
    % =======================================================
    
    patches_path = fullfile(root_path, 'patches');
    
    % Check if path exists
    if ~exist(patches_path, 'dir')
        fprintf('ERROR: Path does not exist: %s\n', patches_path);
        fprintf('Current directory: %s\n', pwd);
        fprintf('Please make sure raw_data/MARIDA/patches exists\n');
        return;
    end
    
    % Get all scene subdirectories
    scene_dirs = dir(patches_path);
    scene_dirs = scene_dirs([scene_dirs.isdir]);
    scene_dirs = scene_dirs(~ismember({scene_dirs.name}, {'.', '..'}));
    
    fprintf('Found %d scene directories\n', length(scene_dirs));
    
    % Initialize lists
    normal_files = {};
    anomaly_files = {};
    
    % Statistics
    total_checked = 0;
    num_normal = 0;
    num_anomaly = 0;
    
    fprintf('\nScanning images...\n');
    
    % Loop through each scene folder
    for s = 1:length(scene_dirs)
        scene_path = fullfile(patches_path, scene_dirs(s).name);
        
        % Find all mask files (_cl.tif)
        mask_files = dir(fullfile(scene_path, '*_cl.tif'));
        
        for f = 1:length(mask_files)
            mask_filename = mask_files(f).name;
            total_checked = total_checked + 1;
            
            % Get corresponding image filename
            basename = strrep(mask_filename, '_cl.tif', '');
            img_filename = [basename '.tif'];
            img_path = fullfile(scene_path, img_filename);
            
            % Skip if image doesn't exist
            if ~exist(img_path, 'file')
                continue;
            end
            
            % Read the mask
            mask = imread(fullfile(scene_path, mask_filename));
            
            % Check for anomaly (debris = 1, ship = 5)
            has_debris = any(mask(:) == 1);
            has_ship = any(mask(:) == 5);
            is_anomaly = has_debris || has_ship;
            
            % Check for water (class 7)
            has_water = any(mask(:) == 7);
            
            % Classify
            if is_anomaly
                anomaly_files{end+1} = img_path;
                num_anomaly = num_anomaly + 1;
            elseif has_water
                normal_files{end+1} = img_path;
                num_normal = num_normal + 1;
            end
            % Images with neither water nor anomaly are ignored
            
            % Progress indicator
            if mod(total_checked, 50) == 0
                fprintf('  Processed %d masks... (Normal: %d, Anomaly: %d)\n', ...
                    total_checked, num_normal, num_anomaly);
            end
        end
    end
    
    % Print summary
    fprintf('\n========== SCAN COMPLETE ==========\n');
    fprintf('Total masks checked: %d\n', total_checked);
    fprintf('NORMAL images found: %d\n', num_normal);
    fprintf('ANOMALY images found: %d\n', num_anomaly);
    
    % Check if we have enough data
    if num_normal == 0
        fprintf('\nERROR: No NORMAL images found!\n');
        fprintf('Please check if the dataset contains water class (7)\n');
        return;
    end
    
    if num_anomaly == 0
        fprintf('\nERROR: No ANOMALY images found!\n');
        fprintf('Please check if the dataset contains debris (1) or ship (5)\n');
        return;
    end
    
    % Balance dataset (take minimum of both classes)
    num_samples = min(num_normal, num_anomaly);
    
    % Optional: Limit to 200 samples per class for faster processing
    if num_samples > 200
        num_samples = 200;
        fprintf('\nLimiting to %d samples per class for faster processing\n', num_samples);
    end
    
    normal_files = normal_files(1:num_samples);
    anomaly_files = anomaly_files(1:num_samples);
    
    fprintf('\nAfter balancing: %d NORMAL, %d ANOMALY\n', ...
        length(normal_files), length(anomaly_files));
    
    % Create data folder if not exists
    if ~exist('../data', 'dir')
        mkdir('../data');
    end
    
    % Save file lists to data folder
    save('../data/file_lists.mat', 'normal_files', 'anomaly_files');
    fprintf('\nSaved ../data/file_lists.mat\n');
    
    % Display sample paths for verification
    fprintf('\nSample NORMAL image:\n  %s\n', normal_files{1});
    fprintf('Sample ANOMALY image:\n  %s\n', anomaly_files{1});
    
end
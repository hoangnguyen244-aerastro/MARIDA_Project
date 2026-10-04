%% step3_train_svm.m (v5 - official MARIDA split, leakage-free)
function step3_train_svm()
    fprintf('\nSTEP 3: Training SVM (official MARIDA split)\n');
    load('../data/features_data.mat','all_features','all_labels','valid_files');

    train_names = read_split('../raw_data/MARIDA/splits/train_X.txt');
    val_names   = read_split('../raw_data/MARIDA/splits/val_X.txt');
    test_names  = read_split('../raw_data/MARIDA/splits/test_X.txt');

    sample_ids = cellfun(@file_id, valid_files, 'UniformOutput', false);
    is_train = ismember(sample_ids, train_names);
    is_val   = ismember(sample_ids, val_names);
    is_test  = ismember(sample_ids, test_names);

    if any((is_train + is_val + is_test) > 1)
        error('A sample occurs in more than one official split.');
    end
    unmatched = ~(is_train | is_val | is_test);
    if any(unmatched)
        fprintf('WARNING: %d valid samples are absent from official split files and are excluded.\n',sum(unmatched));
    end

    dev_idx = find(is_train | is_val);
    test_idx = find(is_test);
    if isempty(dev_idx) || isempty(test_idx), error('Official split mapping produced an empty development or test set.'); end

    X_dev = all_features(dev_idx,:); Y_dev = all_labels(dev_idx);
    X_test = all_features(test_idx,:); Y_test = all_labels(test_idx);

    % Fit preprocessing on development data only.
    mu = mean(X_dev,1); sigma = std(X_dev,0,1); sigma(sigma==0)=1;
    X_dev = bsxfun(@rdivide,bsxfun(@minus,X_dev,mu),sigma);
    X_test = bsxfun(@rdivide,bsxfun(@minus,X_test,mu),sigma);
    X_dev(~isfinite(X_dev))=0; X_test(~isfinite(X_test))=0;

    C_values=[0.1 1 10 100]; scale_values=[0.01 0.1 1 10];
    best_accuracy=-inf; best_C=1; best_scale=1;
    rng(42);
    cvp=cvpartition(Y_dev,'KFold',5);
    for i=1:numel(C_values)
        for j=1:numel(scale_values)
            mdl=fitcsvm(X_dev,Y_dev,'KernelFunction','rbf','BoxConstraint',C_values(i), ...
                'KernelScale',scale_values(j),'Standardize',false);
            cvmdl=crossval(mdl,'CVPartition',cvp);
            acc=1-kfoldLoss(cvmdl);
            fprintf('C=%g, KernelScale=%g -> CV %.2f%%\n',C_values(i),scale_values(j),100*acc);
            if acc>best_accuracy
                best_accuracy=acc; best_C=C_values(i); best_scale=scale_values(j);
            end
        end
    end

    final_svm=fitcsvm(X_dev,Y_dev,'KernelFunction','rbf','BoxConstraint',best_C, ...
        'KernelScale',best_scale,'Standardize',false);
    Y_pred=predict(final_svm,X_test);
    TP=sum(Y_pred==1 & Y_test==1); TN=sum(Y_pred==0 & Y_test==0);
    FP=sum(Y_pred==1 & Y_test==0); FN=sum(Y_pred==0 & Y_test==1);
    fprintf('Official test: n=%d, accuracy=%.2f%%\n',numel(Y_test),100*(TP+TN)/numel(Y_test));
    best_gamma=best_scale; % backward-compatible variable name; this is MATLAB KernelScale, not gamma.
    train_idx=dev_idx;
    save('../models/svm_model.mat','final_svm','mu','sigma','best_C','best_gamma','best_scale','train_idx','test_idx','dev_idx');
end

function names=read_split(path)
    fid=fopen(path,'r'); if fid<0, error('Cannot open split file: %s',path); end
    C=textscan(fid,'%s'); fclose(fid); names=C{1};
end

function id=file_id(path)
    [~,id,~]=fileparts(path);
    % MARIDA image files are named S2_<split-id>.tif, whereas the official
    % split text files store <split-id> without the leading "S2_".
    if length(id) >= 3 && strcmp(id(1:3),'S2_')
        id=id(4:end);
    end
end

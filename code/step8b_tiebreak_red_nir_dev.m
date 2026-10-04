%% step8b_tiebreak_red_nir_dev.m
% Development-only tie-break between the two top Step-8 representations.
% Uses repeated stratified 5-fold CV. Official test is NEVER accessed.
% Selection rule: higher mean accuracy; if practically tied (<0.5 percentage
% point), prefer Red 665 nm as the simpler visible-band interpretation.
% MATLAB R2015b compatible.

function step8b_tiebreak_red_nir_dev()
    fprintf('\n====================================================\n');
    fprintf('STEP 8B: RED vs NIR TIE-BREAK - DEVELOPMENT ONLY\n');
    fprintf('====================================================\n');

    load('../data/features_data.mat','all_labels','valid_files');
    load('../models/svm_model.mat','dev_idx');
    Y=all_labels(dev_idx); files=valid_files(dev_idx);

    reps={'red_665','nir_842'}; channels=[4 8];
    C=10; scale=1; repeats=20; folds=5;
    out=repmat(struct('representation','','mean_accuracy',NaN,'std_accuracy',NaN, ...
        'min_accuracy',NaN,'max_accuracy',NaN,'repeats',repeats,'folds',folds),1,2);

    for r=1:2
        fprintf('\nExtracting %s...\n',reps{r});
        X=zeros(numel(files),18); valid=true(numel(files),1);
        for i=1:numel(files)
            try
                I=double(imread(files{i}));
                X(i,:)=dwt18(I(:,:,channels(r)),'db4');
                if any(~isfinite(X(i,:))), valid(i)=false; end
            catch
                valid(i)=false;
            end
        end
        X=X(valid,:); Yr=Y(valid);
        X=sign(X).*log1p(abs(X));

        acc=zeros(repeats,1);
        for q=1:repeats
            rng(1000+q);
            cvp=cvpartition(Yr,'KFold',folds);
            pred=zeros(size(Yr));
            for f=1:folds
                tr=training(cvp,f); va=test(cvp,f);
                mu=mean(X(tr,:),1); sg=std(X(tr,:),0,1); sg(sg==0)=1;
                A=bsxfun(@rdivide,bsxfun(@minus,X(tr,:),mu),sg);
                B=bsxfun(@rdivide,bsxfun(@minus,X(va,:),mu),sg);
                A(~isfinite(A))=0; B(~isfinite(B))=0;
                m=fitcsvm(A,Yr(tr),'KernelFunction','rbf','BoxConstraint',C, ...
                    'KernelScale',scale,'Standardize',false);
                pred(va)=predict(m,B);
            end
            acc(q)=mean(pred==Yr);
        end
        out(r)=struct('representation',reps{r},'mean_accuracy',mean(acc), ...
            'std_accuracy',std(acc),'min_accuracy',min(acc),'max_accuracy',max(acc), ...
            'repeats',repeats,'folds',folds);
        fprintf('%s: %.3f%% +/- %.3f%%\n',reps{r},100*mean(acc),100*std(acc));
    end

    delta=out(1).mean_accuracy-out(2).mean_accuracy;
    if abs(delta)<0.005
        selected='red_665';
        reason='Mean CV accuracies differ by <0.5 percentage point; predeclared simplicity tie-break selects Red 665 nm.';
    elseif delta>0
        selected='red_665'; reason='Higher repeated-CV mean accuracy.';
    else
        selected='nir_842'; reason='Higher repeated-CV mean accuracy.';
    end

    save('../results/spectral_tiebreak_dev.mat','out','selected','reason','delta');
    fid=fopen('../results/spectral_tiebreak_dev.csv','w');
    fprintf(fid,'Representation,Mean_Accuracy,Std_Accuracy,Min_Accuracy,Max_Accuracy,Repeats,Folds\n');
    for r=1:2
        fprintf(fid,'%s,%.6f,%.6f,%.6f,%.6f,%d,%d\n',out(r).representation, ...
            out(r).mean_accuracy,out(r).std_accuracy,out(r).min_accuracy,out(r).max_accuracy,repeats,folds);
    end
    fprintf(fid,'SELECTED,%s,,,,,\n',selected);
    fclose(fid);

    fprintf('\nLOCKED DEVELOPMENT SELECTION: %s\n%s\n',selected,reason);
    fprintf('Commit the two Step-8B result files BEFORE any final test evaluation.\n');
end

function f=dwt18(J,w)
    [C,S]=wavedec2(J,2,w);
    [H1,V1,D1]=detcoef2('all',C,S,1); [H2,V2,D2]=detcoef2('all',C,S,2);
    cc={H1,V1,D1,H2,V2,D2}; f=zeros(1,18); z=1;
    for k=1:6
        v=cc{k}(:); e=sum(v.^2); den=sum(abs(v));
        if den==0, ent=0; else p=abs(v)/den; p(p==0)=[]; ent=-sum(p.*log2(p)); end
        s=std(v); if numel(v)<3 || s==0, sk=0; else sk=(sum((v-mean(v)).^3)/numel(v))/(s^3); end
        f(z:z+2)=[e ent sk]; z=z+3;
    end
end

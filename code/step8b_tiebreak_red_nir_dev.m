%% step8b_tiebreak_dev.m
% Generic development-only tie-break after Step 8.
% Reads the NEW natural-distribution Step-8 results, finds the two best
% RBF-SVM representations, and uses repeated stratified 5-fold CV with each
% candidate's already-selected hyperparameters. Official test is never used.
% If the Step-8 gap is >=0.5 percentage point, no tie-break is needed.
% MATLAB R2015b compatible.

function step8b_tiebreak_red_nir_dev()
    fprintf('\n====================================================\n');
    fprintf('STEP 8B: GENERIC DEVELOPMENT-ONLY TIE-BREAK\n');
    fprintf('====================================================\n');

    load('../data/features_data.mat','all_labels','valid_files');
    load('../models/svm_model.mat','dev_idx');
    S=load('../results/spectral_ablation_dev.mat','out');
    out8=S.out;

    scores=[out8.svm_cv_accuracy];
    [~,ord]=sort(scores,'descend');
    a=ord(1); b=ord(2);
    gap=scores(a)-scores(b);
    fprintf('Top Step-8 candidates: %s %.3f%% vs %s %.3f%% (gap %.3f pp)\n', ...
        out8(a).representation,100*scores(a),out8(b).representation,100*scores(b),100*gap);

    if gap>=0.005
        selected=out8(a).representation;
        reason='Step-8 CV gap is at least 0.5 percentage point; no repeated-CV tie-break required.';
        out=[];
        save('../results/spectral_tiebreak_dev.mat','out','selected','reason','gap');
        write_selection(selected,reason);
        fprintf('LOCKED DEVELOPMENT SELECTION: %s\n',selected);
        return;
    end

    Y=all_labels(dev_idx); files=valid_files(dev_idx);
    n_dev=numel(Y); n_normal=sum(Y==0); n_anomaly=sum(Y==1);
    cand=[a b]; repeats=20; folds=5;
    out=repmat(struct('representation','','mean_accuracy',NaN,'std_accuracy',NaN, ...
        'min_accuracy',NaN,'max_accuracy',NaN,'C',NaN,'KernelScale',NaN, ...
        'valid_dev',0,'repeats',repeats,'folds',folds),1,2);

    for z=1:2
        q=out8(cand(z)); rep=q.representation;
        fprintf('\nExtracting %s...\n',rep);
        [X,Yr]=extract_rep(files,Y,rep);
        X=sign(X).*log1p(abs(X));
        acc=zeros(repeats,1);
        for rr=1:repeats
            rng(1000+rr); cvp=cvpartition(Yr,'KFold',folds);
            pred=zeros(size(Yr));
            for ff=1:folds
                tr=training(cvp,ff); va=test(cvp,ff);
                mu=mean(X(tr,:),1); sg=std(X(tr,:),0,1); sg(sg==0)=1;
                A=bsxfun(@rdivide,bsxfun(@minus,X(tr,:),mu),sg);
                B=bsxfun(@rdivide,bsxfun(@minus,X(va,:),mu),sg);
                A(~isfinite(A))=0; B(~isfinite(B))=0;
                m=fitcsvm(A,Yr(tr),'KernelFunction','rbf','BoxConstraint',q.svm_C, ...
                    'KernelScale',q.svm_KernelScale,'Standardize',false);
                pred(va)=predict(m,B);
            end
            acc(rr)=mean(pred==Yr);
        end
        out(z)=struct('representation',rep,'mean_accuracy',mean(acc), ...
            'std_accuracy',std(acc),'min_accuracy',min(acc),'max_accuracy',max(acc), ...
            'C',q.svm_C,'KernelScale',q.svm_KernelScale,'valid_dev',numel(Yr), ...
            'repeats',repeats,'folds',folds);
        fprintf('%s: %.3f%% +/- %.3f%%\n',rep,100*mean(acc),100*std(acc));
    end

    delta=out(1).mean_accuracy-out(2).mean_accuracy;
    if abs(delta)<0.005
        % Predeclared deterministic simplicity rule: lower dimensionality,
        % then Step-8 rank if dimensions are equal.
        d1=out8(cand(1)).dimensions; d2=out8(cand(2)).dimensions;
        if d2<d1, selected=out(2).representation; else selected=out(1).representation; end
        reason='Repeated-CV means differ by <0.5 percentage point; predeclared simplicity/rank tie-break applied.';
    elseif delta>0
        selected=out(1).representation; reason='Higher repeated-CV mean accuracy.';
    else
        selected=out(2).representation; reason='Higher repeated-CV mean accuracy.';
    end

    save('../results/spectral_tiebreak_dev.mat','out','selected','reason','delta', ...
        'n_dev','n_normal','n_anomaly');
    fid=fopen('../results/spectral_tiebreak_dev.csv','w');
    fprintf(fid,'Representation,Valid_Dev,Dev_Normal,Dev_Anomaly,Mean_Accuracy,Std_Accuracy,Min_Accuracy,Max_Accuracy,C,KernelScale,Repeats,Folds\n');
    for z=1:2
        q=out(z);
        fprintf(fid,'%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%g,%g,%d,%d\n', ...
            q.representation,q.valid_dev,n_normal,n_anomaly,q.mean_accuracy,q.std_accuracy, ...
            q.min_accuracy,q.max_accuracy,q.C,q.KernelScale,repeats,folds);
    end
    fprintf(fid,'SELECTED,%s,,,,,,,,,,\n',selected); fclose(fid);
    fprintf('\nLOCKED DEVELOPMENT SELECTION: %s\n%s\n',selected,reason);
end

function write_selection(selected,reason)
    fid=fopen('../results/spectral_tiebreak_dev.csv','w');
    fprintf(fid,'Selection,Reason\n%s,%s\n',selected,reason); fclose(fid);
end

function [X,Yr]=extract_rep(files,Y,rep)
    if strcmp(rep,'red_nir_swir1_54d'), dim=54; else dim=18; end
    F=zeros(numel(files),dim); valid=true(numel(files),1);
    for i=1:numel(files)
        try
            I=double(imread(files{i}));
            if ndims(I)~=3 || size(I,3)~=11, error('Expected MARIDA 11-band TIFF'); end
            switch rep
                case 'legacy_wrong_rgb', J=.2989*I(:,:,1)+.5870*I(:,:,2)+.1140*I(:,:,3); F(i,:)=dwt18(J,'db4');
                case 'true_rgb_gray', J=.2989*I(:,:,4)+.5870*I(:,:,3)+.1140*I(:,:,2); F(i,:)=dwt18(J,'db4');
                case 'red_665', F(i,:)=dwt18(I(:,:,4),'db4');
                case 'nir_842', F(i,:)=dwt18(I(:,:,8),'db4');
                case 'swir1_1600', F(i,:)=dwt18(I(:,:,10),'db4');
                case 'red_nir_swir1_54d', F(i,:)=[dwt18(I(:,:,4),'db4') dwt18(I(:,:,8),'db4') dwt18(I(:,:,10),'db4')];
                otherwise, error('Unknown representation');
            end
            if any(~isfinite(F(i,:))), valid(i)=false; end
        catch, valid(i)=false;
        end
    end
    X=F(valid,:); Yr=Y(valid);
end

function f=dwt18(J,w)
    [C,S]=wavedec2(J,2,w); [H1,V1,D1]=detcoef2('all',C,S,1); [H2,V2,D2]=detcoef2('all',C,S,2);
    cc={H1,V1,D1,H2,V2,D2}; f=zeros(1,18); z=1;
    for k=1:6
        v=cc{k}(:); e=sum(v.^2); den=sum(abs(v));
        if den==0, ent=0; else p=abs(v)/den; p(p==0)=[]; ent=-sum(p.*log2(p)); end
        s=std(v); if numel(v)<3 || s==0, sk=0; else sk=(sum((v-mean(v)).^3)/numel(v))/(s^3); end
        f(z:z+2)=[e ent sk]; z=z+3;
    end
end

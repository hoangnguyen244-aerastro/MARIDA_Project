%% step8_spectral_ablation_dev.m
% Spectral representation ablation using DEVELOPMENT CV ONLY.
% IMPORTANT: This script never evaluates or ranks representations on official test.
% MARIDA official channel mapping:
% 1..11 = 440,490,560,665,705,740,783,842,865,1600,2200 nm.
% MATLAB R2015b compatible.

function step8_spectral_ablation_dev()
    fprintf('\n====================================================\n');
    fprintf('STEP 8: SPECTRAL ABLATION - DEVELOPMENT CV ONLY\n');
    fprintf('====================================================\n');

    load('../data/features_data.mat','all_labels','valid_files');
    load('../models/svm_model.mat','dev_idx');

    reps={'legacy_wrong_rgb','true_rgb_gray','red_665','nir_842','swir1_1600', ...
          'red_nir_swir1_54d'};
    dims=[18 18 18 18 18 54];
    nrep=numel(reps);
    out=repmat(struct('representation','','dimensions',0,'valid_dev',0, ...
        'svm_cv_accuracy',NaN,'svm_C',NaN,'svm_KernelScale',NaN, ...
        'knn_cv_accuracy',NaN,'knn_k',NaN),1,nrep);

    Y=all_labels(dev_idx);
    files=valid_files(dev_idx);
    n_dev=numel(Y); n_normal=sum(Y==0); n_anomaly=sum(Y==1);
    fprintf('Natural development set: N=%d | Normal=%d | Anomaly=%d\n', ...
        n_dev,n_normal,n_anomaly);
    Cvals=[0.1 1 10 100]; scales=[0.01 0.1 1 10]; kvals=[1 3 5 7 9 15];

    for r=1:nrep
        fprintf('\n[%d/%d] %s (%d-D)\n',r,nrep,reps{r},dims(r));
        F=zeros(numel(files),dims(r)); valid=true(numel(files),1);
        for i=1:numel(files)
            try
                F(i,:)=spectral_wavelet_features(files{i},reps{r},'db4');
                if any(~isfinite(F(i,:))), valid(i)=false; end
            catch
                valid(i)=false;
            end
            if mod(i,100)==0, fprintf('  extracted %d/%d\n',i,numel(files)); end
        end
        X=F(valid,:); Yr=Y(valid);
        X=sign(X).*log1p(abs(X));

        % Fold-local preprocessing: normalization is fit inside each training fold.
        rng(42); cvp=cvpartition(Yr,'KFold',5);
        bestS=-inf; bestC=Cvals(1); bestScale=scales(1);
        for a=1:numel(Cvals)
            for b=1:numel(scales)
                acc=cv_svm_foldnorm(X,Yr,cvp,Cvals(a),scales(b));
                if acc>bestS, bestS=acc; bestC=Cvals(a); bestScale=scales(b); end
            end
        end
        bestK=-inf; bk=kvals(1);
        for a=1:numel(kvals)
            acc=cv_knn_foldnorm(X,Yr,cvp,kvals(a));
            if acc>bestK, bestK=acc; bk=kvals(a); end
        end

        out(r)=struct('representation',reps{r},'dimensions',dims(r),'valid_dev',sum(valid), ...
            'svm_cv_accuracy',bestS,'svm_C',bestC,'svm_KernelScale',bestScale, ...
            'knn_cv_accuracy',bestK,'knn_k',bk);
        fprintf('  RBF-SVM CV %.2f%% | k-NN CV %.2f%%\n',100*bestS,100*bestK);
    end

    if ~exist('../results','dir'), mkdir('../results'); end
    save('../results/spectral_ablation_dev.mat','out','n_dev','n_normal','n_anomaly');
    fid=fopen('../results/spectral_ablation_dev.csv','w');
    fprintf(fid,'Representation,Dimensions,Valid_Dev,Dev_Normal,Dev_Anomaly,SVM_CV_Accuracy,SVM_C,SVM_KernelScale,kNN_CV_Accuracy,kNN_k\n');
    for r=1:nrep
        q=out(r);
        fprintf(fid,'%s,%d,%d,%d,%d,%.6f,%g,%g,%.6f,%d\n',q.representation,q.dimensions, ...
            q.valid_dev,n_normal,n_anomaly,q.svm_cv_accuracy,q.svm_C,q.svm_KernelScale,q.knn_cv_accuracy,q.knn_k);
    end
    fclose(fid);
    fprintf('\nSaved development-only ablation results. DO NOT select using official test.\n');
end

function f=spectral_wavelet_features(path,rep,w)
    I=double(imread(path));
    if ndims(I)~=3 || size(I,3)~=11, error('Expected MARIDA 11-band TIFF'); end
    switch rep
        case 'legacy_wrong_rgb'
            J=0.2989*I(:,:,1)+0.5870*I(:,:,2)+0.1140*I(:,:,3);
            f=dwt18(J,w);
        case 'true_rgb_gray'
            % Correct physical RGB: R=665nm(ch4), G=560nm(ch3), B=490nm(ch2).
            J=0.2989*I(:,:,4)+0.5870*I(:,:,3)+0.1140*I(:,:,2);
            f=dwt18(J,w);
        case 'red_665'
            f=dwt18(I(:,:,4),w);
        case 'nir_842'
            f=dwt18(I(:,:,8),w);
        case 'swir1_1600'
            f=dwt18(I(:,:,10),w);
        case 'red_nir_swir1_54d'
            f=[dwt18(I(:,:,4),w),dwt18(I(:,:,8),w),dwt18(I(:,:,10),w)];
        otherwise
            error('Unknown representation');
    end
end

function f=dwt18(J,w)
    [C,S]=wavedec2(J,2,w);
    [H1,V1,D1]=detcoef2('all',C,S,1); [H2,V2,D2]=detcoef2('all',C,S,2);
    cc={H1,V1,D1,H2,V2,D2}; f=zeros(1,18); z=1;
    for k=1:6
        v=cc{k}(:); e=sum(v.^2);
        den=sum(abs(v));
        if den==0, ent=0; else p=abs(v)/den; p(p==0)=[]; ent=-sum(p.*log2(p)); end
        s=std(v); if numel(v)<3 || s==0, sk=0; else sk=(sum((v-mean(v)).^3)/numel(v))/(s^3); end
        f(z:z+2)=[e ent sk]; z=z+3;
    end
end

function acc=cv_svm_foldnorm(X,Y,cvp,C,scale)
    pred=zeros(size(Y));
    for fold=1:cvp.NumTestSets
        tr=training(cvp,fold); va=test(cvp,fold);
        mu=mean(X(tr,:),1); sg=std(X(tr,:),0,1); sg(sg==0)=1;
        A=bsxfun(@rdivide,bsxfun(@minus,X(tr,:),mu),sg);
        B=bsxfun(@rdivide,bsxfun(@minus,X(va,:),mu),sg);
        A(~isfinite(A))=0; B(~isfinite(B))=0;
        m=fitcsvm(A,Y(tr),'KernelFunction','rbf','BoxConstraint',C,'KernelScale',scale,'Standardize',false);
        pred(va)=predict(m,B);
    end
    acc=mean(pred==Y);
end

function acc=cv_knn_foldnorm(X,Y,cvp,k)
    pred=zeros(size(Y));
    for fold=1:cvp.NumTestSets
        tr=training(cvp,fold); va=test(cvp,fold);
        mu=mean(X(tr,:),1); sg=std(X(tr,:),0,1); sg(sg==0)=1;
        A=bsxfun(@rdivide,bsxfun(@minus,X(tr,:),mu),sg);
        B=bsxfun(@rdivide,bsxfun(@minus,X(va,:),mu),sg);
        A(~isfinite(A))=0; B(~isfinite(B))=0;
        m=fitcknn(A,Y(tr),'NumNeighbors',k,'Standardize',false);
        pred(va)=predict(m,B);
    end
    acc=mean(pred==Y);
end

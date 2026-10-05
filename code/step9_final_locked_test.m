%% step9_final_locked_test.m
% FINAL locked evaluation on the official MARIDA test set.
% Representation and SVM hyperparameters are read from development-only
% Step 8/8B outputs. No model/representation selection is performed here.
% MATLAB R2015b compatible.

function step9_final_locked_test()
    fprintf('\n====================================================\n');
    fprintf('STEP 9: FINAL LOCKED OFFICIAL-TEST EVALUATION\n');
    fprintf('====================================================\n');

    load('../data/features_data.mat','all_labels','valid_files');
    load('../models/svm_model.mat','dev_idx','test_idx');
    T=load('../results/spectral_tiebreak_dev.mat','selected');
    S=load('../results/spectral_ablation_dev.mat','out');
    representation=T.selected;

    names={S.out.representation}; k=find(strcmp(names,representation),1);
    if isempty(k), error('Locked representation not found in Step-8 results.'); end
    C=S.out(k).svm_C; scale=S.out(k).svm_KernelScale;
    fprintf('LOCKED: %s | db4 | %d-D | RBF-SVM C=%g KernelScale=%g\n', ...
        representation,S.out(k).dimensions,C,scale);

    [Xdev,Ydev,dev_files]=extract_set(valid_files(dev_idx),all_labels(dev_idx),representation);
    [Xtest,Ytest,test_files]=extract_set(valid_files(test_idx),all_labels(test_idx),representation);
    Xdev=sign(Xdev).*log1p(abs(Xdev)); Xtest=sign(Xtest).*log1p(abs(Xtest));

    mu=mean(Xdev,1); sigma=std(Xdev,0,1); sigma(sigma==0)=1;
    A=bsxfun(@rdivide,bsxfun(@minus,Xdev,mu),sigma);
    B=bsxfun(@rdivide,bsxfun(@minus,Xtest,mu),sigma);
    A(~isfinite(A))=0; B(~isfinite(B))=0;

    final_model=fitcsvm(A,Ydev,'KernelFunction','rbf','BoxConstraint',C, ...
        'KernelScale',scale,'Standardize',false);
    [pred,score]=predict(final_model,B);

    TN=sum((Ytest==0)&(pred==0)); FP=sum((Ytest==0)&(pred==1));
    FN=sum((Ytest==1)&(pred==0)); TP=sum((Ytest==1)&(pred==1));
    accuracy=(TP+TN)/numel(Ytest); precision=safe_div(TP,TP+FP);
    recall=safe_div(TP,TP+FN); specificity=safe_div(TN,TN+FP);
    f1=safe_div(2*precision*recall,precision+recall);
    balanced_accuracy=(recall+specificity)/2;

    auc=NaN; fpr=[]; tpr=[];
    poscol=find(final_model.ClassNames==1,1);
    if ~isempty(poscol)
        try, [fpr,tpr,~,auc]=perfcurve(Ytest,score(:,poscol),1); catch, end
    end

    n_test=numel(Ytest); test_normal=sum(Ytest==0); test_anomaly=sum(Ytest==1);
    n_dev=numel(Ydev); dev_normal=sum(Ydev==0); dev_anomaly=sum(Ydev==1);
    fprintf('\nDevelopment: N=%d | Normal=%d | Anomaly=%d\n',n_dev,dev_normal,dev_anomaly);
    fprintf('Official test: N=%d | Normal=%d | Anomaly=%d\n',n_test,test_normal,test_anomaly);
    fprintf('Confusion: TN=%d FP=%d FN=%d TP=%d\n',TN,FP,FN,TP);
    fprintf('Accuracy: %.2f%% | Balanced accuracy: %.2f%%\n',100*accuracy,100*balanced_accuracy);
    fprintf('Precision: %.4f | Recall: %.4f | Specificity: %.4f | F1: %.4f | AUC: %.4f\n', ...
        precision,recall,specificity,f1,auc);

    save('../results/final_locked_test.mat','representation','accuracy','balanced_accuracy', ...
        'precision','recall','specificity','f1','auc','TN','FP','FN','TP','fpr','tpr', ...
        'Ytest','pred','score','test_files','dev_files','mu','sigma','C','scale', ...
        'n_dev','dev_normal','dev_anomaly','n_test','test_normal','test_anomaly');

    fid=fopen('../results/final_locked_test.csv','w');
    fprintf(fid,'Representation,Wavelet,Dimensions,Classifier,C,KernelScale,N_Dev,Dev_Normal,Dev_Anomaly,N_Test,Test_Normal,Test_Anomaly,TN,FP,FN,TP,Accuracy,Balanced_Accuracy,Precision,Recall,Specificity,F1,AUC\n');
    fprintf(fid,'%s,db4,%d,RBF-SVM,%g,%g,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
        representation,S.out(k).dimensions,C,scale,n_dev,dev_normal,dev_anomaly,n_test,test_normal,test_anomaly, ...
        TN,FP,FN,TP,accuracy,balanced_accuracy,precision,recall,specificity,f1,auc);
    fclose(fid);

    save('../models/final_locked_svm.mat','final_model','mu','sigma','C','scale','representation');
    fprintf('\nSaved final official-test results and locked deployment model.\n');
end

function [X,Y,files_out]=extract_set(files,Yin,rep)
    if strcmp(rep,'red_nir_swir1_54d'), dim=54; else dim=18; end
    X=zeros(numel(files),dim); valid=true(numel(files),1);
    for i=1:numel(files)
        try
            I=double(imread(files{i}));
            if ndims(I)~=3 || size(I,3)~=11, error('Expected 11-band MARIDA TIFF'); end
            switch rep
                case 'legacy_wrong_rgb', J=.2989*I(:,:,1)+.5870*I(:,:,2)+.1140*I(:,:,3); X(i,:)=dwt18(J,'db4');
                case 'true_rgb_gray', J=.2989*I(:,:,4)+.5870*I(:,:,3)+.1140*I(:,:,2); X(i,:)=dwt18(J,'db4');
                case 'red_665', X(i,:)=dwt18(I(:,:,4),'db4');
                case 'nir_842', X(i,:)=dwt18(I(:,:,8),'db4');
                case 'swir1_1600', X(i,:)=dwt18(I(:,:,10),'db4');
                case 'red_nir_swir1_54d', X(i,:)=[dwt18(I(:,:,4),'db4') dwt18(I(:,:,8),'db4') dwt18(I(:,:,10),'db4')];
                otherwise, error('Unknown locked representation');
            end
            if any(~isfinite(X(i,:))), valid(i)=false; end
        catch, valid(i)=false;
        end
    end
    X=X(valid,:); Y=Yin(valid); files_out=files(valid);
    if any(~valid), fprintf('Excluded %d invalid samples from this split.\n',sum(~valid)); end
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

function z=safe_div(a,b)
    if b==0, z=0; else z=a/b; end
end

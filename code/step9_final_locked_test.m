%% step9_final_locked_test.m
% FINAL locked evaluation on the official MARIDA test set.
% Configuration was selected using development data only:
% Red 665 nm (channel 4), db4, 18 DWT statistics, RBF-SVM C=10,
% KernelScale=1. No model/representation selection is performed here.
% MATLAB R2015b compatible.

function step9_final_locked_test()
    fprintf('\n====================================================\n');
    fprintf('STEP 9: FINAL LOCKED OFFICIAL-TEST EVALUATION\n');
    fprintf('====================================================\n');

    load('../data/features_data.mat','all_labels','valid_files');
    load('../models/svm_model.mat','dev_idx','test_idx');

    C=10; scale=1;
    fprintf('LOCKED: Red 665 nm | db4 | 18-D | RBF-SVM C=%g KernelScale=%g\n',C,scale);

    [Xdev,Ydev,dev_files]=extract_set(valid_files(dev_idx),all_labels(dev_idx));
    [Xtest,Ytest,test_files]=extract_set(valid_files(test_idx),all_labels(test_idx));

    Xdev=sign(Xdev).*log1p(abs(Xdev));
    Xtest=sign(Xtest).*log1p(abs(Xtest));

    mu=mean(Xdev,1); sigma=std(Xdev,0,1); sigma(sigma==0)=1;
    A=bsxfun(@rdivide,bsxfun(@minus,Xdev,mu),sigma);
    B=bsxfun(@rdivide,bsxfun(@minus,Xtest,mu),sigma);
    A(~isfinite(A))=0; B(~isfinite(B))=0;

    final_model=fitcsvm(A,Ydev,'KernelFunction','rbf','BoxConstraint',C, ...
        'KernelScale',scale,'Standardize',false);
    [pred,score]=predict(final_model,B);

    TN=sum((Ytest==0)&(pred==0)); FP=sum((Ytest==0)&(pred==1));
    FN=sum((Ytest==1)&(pred==0)); TP=sum((Ytest==1)&(pred==1));
    accuracy=(TP+TN)/numel(Ytest);
    precision=safe_div(TP,TP+FP);
    recall=safe_div(TP,TP+FN);
    specificity=safe_div(TN,TN+FP);
    f1=safe_div(2*precision*recall,precision+recall);
    balanced_accuracy=(recall+specificity)/2;

    auc=NaN; fpr=[]; tpr=[];
    try
        [fpr,tpr,~,auc]=perfcurve(Ytest,score(:,2),1);
    catch
    end

    fprintf('\nOfficial test samples used: %d\n',numel(Ytest));
    fprintf('Confusion: TN=%d FP=%d FN=%d TP=%d\n',TN,FP,FN,TP);
    fprintf('Accuracy:          %.2f%%\n',100*accuracy);
    fprintf('Balanced accuracy: %.2f%%\n',100*balanced_accuracy);
    fprintf('Precision:         %.4f\n',precision);
    fprintf('Recall:            %.4f\n',recall);
    fprintf('Specificity:       %.4f\n',specificity);
    fprintf('F1:                %.4f\n',f1);
    fprintf('AUC:               %.4f\n',auc);

    save('../results/final_locked_test.mat','accuracy','balanced_accuracy','precision', ...
        'recall','specificity','f1','auc','TN','FP','FN','TP','fpr','tpr', ...
        'Ytest','pred','score','test_files','dev_files','mu','sigma','C','scale');

    fid=fopen('../results/final_locked_test.csv','w');
    fprintf(fid,'Representation,Wavelet,Dimensions,Classifier,C,KernelScale,N_Test,TN,FP,FN,TP,Accuracy,Balanced_Accuracy,Precision,Recall,Specificity,F1,AUC\n');
    fprintf(fid,'red_665,db4,18,RBF-SVM,%g,%g,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
        C,scale,numel(Ytest),TN,FP,FN,TP,accuracy,balanced_accuracy,precision,recall,specificity,f1,auc);
    fclose(fid);

    save('../models/final_locked_red665_svm.mat','final_model','mu','sigma','C','scale');
    fprintf('\nSaved final test results and locked deployment model.\n');
end

function [X,Y,files_out]=extract_set(files,Yin)
    X=zeros(numel(files),18); valid=true(numel(files),1);
    for i=1:numel(files)
        try
            I=double(imread(files{i}));
            if ndims(I)~=3 || size(I,3)~=11, error('Expected 11-band MARIDA TIFF'); end
            X(i,:)=dwt18(I(:,:,4),'db4');
            if any(~isfinite(X(i,:))), valid(i)=false; end
        catch
            valid(i)=false;
        end
    end
    X=X(valid,:); Y=Yin(valid); files_out=files(valid);
    if any(~valid), fprintf('Excluded %d invalid samples from this split.\n',sum(~valid)); end
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

function z=safe_div(a,b)
    if b==0, z=0; else z=a/b; end
end

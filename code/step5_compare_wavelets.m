%% step5_compare_wavelets.m (v5 - official split, no test-set tuning)
function step5_compare_wavelets()
    load('../data/file_lists.mat','normal_files','anomaly_files');
    wavelets={'db4','sym4','bior3.5','coif2','haar'};
    files=[normal_files(:); anomaly_files(:)];
    labels=[zeros(numel(normal_files),1); ones(numel(anomaly_files),1)];
    ids=cellfun(@file_id,files,'UniformOutput',false);
    tr=read_split('../raw_data/MARIDA/splits/train_X.txt');
    va=read_split('../raw_data/MARIDA/splits/val_X.txt');
    te=read_split('../raw_data/MARIDA/splits/test_X.txt');
    is_dev=ismember(ids,tr)|ismember(ids,va); is_test=ismember(ids,te);
    if any((ismember(ids,tr)+ismember(ids,va)+ismember(ids,te))>1), error('Overlapping official splits.'); end

    C_values=[0.1 1 10 100]; scale_values=[0.01 0.1 1 10];
    results=repmat(struct('wavelet','','accuracy',NaN,'precision',NaN, ...
        'recall',NaN,'f1',NaN,'auc',NaN,'best_C',NaN, ...
        'best_KernelScale',NaN,'cv_accuracy',NaN,'n_dev',0,'n_test',0), ...
        1,numel(wavelets));
    for w=1:numel(wavelets)
        F=zeros(numel(files),18); valid=true(numel(files),1);
        for i=1:numel(files)
            try
                x=wavelet_feature_extractor(files{i},wavelets{w});
                x=sign(x).*log1p(abs(x));
                if any(~isfinite(x)) || ~any(x~=0), valid(i)=false; else, F(i,:)=x; end
            catch
                valid(i)=false;
            end
        end
        dev=find(is_dev & valid); tst=find(is_test & valid);
        Xd=F(dev,:); Yd=labels(dev); Xt=F(tst,:); Yt=labels(tst);
        mu=mean(Xd,1); sd=std(Xd,0,1); sd(sd==0)=1;
        Xd=bsxfun(@rdivide,bsxfun(@minus,Xd,mu),sd);
        Xt=bsxfun(@rdivide,bsxfun(@minus,Xt,mu),sd);
        rng(42); cvp=cvpartition(Yd,'KFold',5);
        best=-inf; bc=1; bs=1;
        for ci=1:numel(C_values)
            for si=1:numel(scale_values)
                m=fitcsvm(Xd,Yd,'KernelFunction','rbf','BoxConstraint',C_values(ci), ...
                    'KernelScale',scale_values(si),'Standardize',false);
                a=1-kfoldLoss(crossval(m,'CVPartition',cvp));
                if a>best, best=a; bc=C_values(ci); bs=scale_values(si); end
            end
        end
        m=fitcsvm(Xd,Yd,'KernelFunction','rbf','BoxConstraint',bc,'KernelScale',bs,'Standardize',false);
        [yp,score]=predict(m,Xt);
        acc=mean(yp==Yt);
        TP=sum(yp==1 & Yt==1); FP=sum(yp==1 & Yt==0); FN=sum(yp==0 & Yt==1);
        precision=TP/max(TP+FP,1); recall=TP/max(TP+FN,1);
        f1=2*precision*recall/max(precision+recall,eps);
        [~,~,~,auc]=perfcurve(Yt,score(:,2),1);
        results(w)=struct('wavelet',wavelets{w},'accuracy',acc,'precision',precision, ...
            'recall',recall,'f1',f1,'auc',auc,'best_C',bc,'best_KernelScale',bs, ...
            'cv_accuracy',best,'n_dev',numel(Yd),'n_test',numel(Yt));
        fprintf('%s: CV %.2f%%; TEST acc %.2f%% F1 %.4f AUC %.4f\n',wavelets{w},100*best,100*acc,f1,auc);
    end
    save('../results/wavelet_comparison.mat','results');
    fid=fopen('../results/wavelet_comparison.csv','w');
    fprintf(fid,'Wavelet,Accuracy,Precision,Recall,F1,AUC,C,KernelScale,CV_Accuracy,N_Dev,N_Test\n');
    for w=1:numel(results)
        r=results(w); fprintf(fid,'%s,%.6f,%.6f,%.6f,%.6f,%.6f,%g,%g,%.6f,%d,%d\n', ...
            r.wavelet,r.accuracy,r.precision,r.recall,r.f1,r.auc,r.best_C,r.best_KernelScale,r.cv_accuracy,r.n_dev,r.n_test);
    end
    fclose(fid);
end
function names=read_split(path)
    fid=fopen(path,'r'); if fid<0,error('Cannot open %s',path);end
    C=textscan(fid,'%s'); fclose(fid); names=C{1};
end
function id=file_id(path)
    [~,id,~]=fileparts(path);
    if length(id) >= 3 && strcmp(id(1:3),'S2_'), id=id(4:end); end
end

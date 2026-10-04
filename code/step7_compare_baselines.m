%% step7_compare_baselines.m
% Classifier baselines on the SAME db4 18-D representation and official split.
% Hyperparameters are selected on development data only; test is evaluated once.
% MATLAB R2015b compatible.

function step7_compare_baselines()
    fprintf('\n========================================\n');
    fprintf('STEP 7: Lightweight classifier baselines\n');
    fprintf('========================================\n');

    load('../data/features_data.mat','all_features','all_labels');
    load('../models/svm_model.mat','dev_idx','test_idx','mu','sigma','best_C','best_scale');

    Xd=all_features(dev_idx,:); Yd=all_labels(dev_idx);
    Xt=all_features(test_idx,:); Yt=all_labels(test_idx);
    Xd=bsxfun(@rdivide,bsxfun(@minus,Xd,mu),sigma);
    Xt=bsxfun(@rdivide,bsxfun(@minus,Xt,mu),sigma);
    Xd(~isfinite(Xd))=0; Xt(~isfinite(Xt))=0;

    names={'RBF-SVM','Linear-SVM','k-NN','Decision-Tree'};
    n=numel(names);
    out=repmat(struct('model','','cv_accuracy',NaN,'accuracy',NaN,'precision',NaN, ...
        'recall',NaN,'f1',NaN,'auc',NaN,'setting',''),1,n);
    rng(42); cvp=cvpartition(Yd,'KFold',5);

    % 1) Main RBF-SVM: use parameters already selected in Step 3.
    m=fitcsvm(Xd,Yd,'KernelFunction','rbf','BoxConstraint',best_C, ...
        'KernelScale',best_scale,'Standardize',false);
    cvacc=1-kfoldLoss(crossval(m,'CVPartition',cvp));
    out(1)=evaluate_model(m,Xt,Yt,names{1},cvacc,sprintf('C=%g; KernelScale=%g',best_C,best_scale));

    % 2) Linear SVM: tune C on development CV.
    Cvals=[0.1 1 10 100]; best=-inf; bc=Cvals(1);
    for k=1:numel(Cvals)
        q=fitcsvm(Xd,Yd,'KernelFunction','linear','BoxConstraint',Cvals(k),'Standardize',false);
        a=1-kfoldLoss(crossval(q,'CVPartition',cvp));
        if a>best, best=a; bc=Cvals(k); end
    end
    m=fitcsvm(Xd,Yd,'KernelFunction','linear','BoxConstraint',bc,'Standardize',false);
    out(2)=evaluate_model(m,Xt,Yt,names{2},best,sprintf('C=%g',bc));

    % 3) k-NN: tune odd k values on development CV.
    kvals=[1 3 5 7 9 15]; best=-inf; bk=kvals(1);
    for k=1:numel(kvals)
        q=fitcknn(Xd,Yd,'NumNeighbors',kvals(k),'Standardize',false);
        a=1-kfoldLoss(crossval(q,'CVPartition',cvp));
        if a>best, best=a; bk=kvals(k); end
    end
    m=fitcknn(Xd,Yd,'NumNeighbors',bk,'Standardize',false);
    out(3)=evaluate_model(m,Xt,Yt,names{3},best,sprintf('k=%d',bk));

    % 4) Single decision tree: tune minimum leaf size on development CV.
    leaves=[1 5 10 20 40]; best=-inf; bl=leaves(1);
    for k=1:numel(leaves)
        q=fitctree(Xd,Yd,'MinLeafSize',leaves(k));
        a=1-kfoldLoss(crossval(q,'CVPartition',cvp));
        if a>best, best=a; bl=leaves(k); end
    end
    m=fitctree(Xd,Yd,'MinLeafSize',bl);
    out(4)=evaluate_model(m,Xt,Yt,names{4},best,sprintf('MinLeafSize=%d',bl));

    if ~exist('../results','dir'), mkdir('../results'); end
    save('../results/baseline_comparison.mat','out');
    fid=fopen('../results/baseline_comparison.csv','w');
    fprintf(fid,'Model,CV_Accuracy,Test_Accuracy,Precision,Recall,F1,AUC,Setting\n');
    for i=1:n
        r=out(i); fprintf(fid,'%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n', ...
            r.model,r.cv_accuracy,r.accuracy,r.precision,r.recall,r.f1,r.auc,r.setting);
        fprintf('%s: CV %.2f%% | TEST acc %.2f%% | F1 %.4f | AUC %.4f | %s\n', ...
            r.model,100*r.cv_accuracy,100*r.accuracy,r.f1,r.auc,r.setting);
    end
    fclose(fid);
end

function r=evaluate_model(m,X,Y,name,cvacc,setting)
    [yp,s]=predict(m,X);
    TP=sum(yp==1 & Y==1); FP=sum(yp==1 & Y==0); FN=sum(yp==0 & Y==1); TN=sum(yp==0 & Y==0);
    acc=(TP+TN)/numel(Y); p=TP/max(TP+FP,1); rec=TP/max(TP+FN,1);
    f=2*p*rec/max(p+rec,eps);
    auc=NaN;
    if size(s,2)>=2
        [~,~,~,auc]=perfcurve(Y,s(:,2),1);
    end
    r=struct('model',name,'cv_accuracy',cvacc,'accuracy',acc,'precision',p, ...
        'recall',rec,'f1',f,'auc',auc,'setting',setting);
end

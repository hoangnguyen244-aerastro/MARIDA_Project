%% step7_compare_baselines.m
% DEVELOPMENT-ONLY classifier comparison on the locked NIR-842/db4/18-D representation.
% No official-test labels or predictions are accessed here.
% MATLAB R2015b compatible.
function step7_compare_baselines()
fprintf('\nSTEP 7: NIR-842 CLASSIFIER BASELINES - DEVELOPMENT CV ONLY\n');
load('../data/features_data.mat','all_labels','valid_files');
load('../models/svm_model.mat','dev_idx');
S=load('../results/spectral_ablation_dev.mat','out');
T=load('../results/spectral_tiebreak_dev.mat','selected');
rep=T.selected; names8={S.out.representation}; z=find(strcmp(names8,rep),1);
if isempty(z), error('Locked representation missing from Step 8.'); end
Cmain=S.out(z).svm_C; scale=S.out(z).svm_KernelScale;
if ~strcmp(rep,'nir_842'), error('This finalized baseline script expects locked NIR-842.'); end
Y=all_labels(dev_idx); files=valid_files(dev_idx);
X=zeros(numel(files),18); valid=true(numel(files),1);
for i=1:numel(files)
 try
  I=double(imread(files{i})); X(i,:)=dwt18(I(:,:,8),'db4');
  if any(~isfinite(X(i,:))), valid(i)=false; end
 catch, valid(i)=false;
 end
end
X=X(valid,:); Y=Y(valid); X=sign(X).*log1p(abs(X));
rng(42); cvp=cvpartition(Y,'KFold',5);
models={'RBF-SVM','Linear-SVM','k-NN','Decision-Tree'};
out=repmat(struct('model','','cv_accuracy',NaN,'setting',''),1,4);
out(1)=struct('model',models{1},'cv_accuracy',cv_svm(X,Y,cvp,'rbf',Cmain,scale),'setting',sprintf('C=%g; KernelScale=%g',Cmain,scale));
Cvals=[.1 1 10 100]; best=-inf; bc=Cvals(1);
for i=1:numel(Cvals), a=cv_svm(X,Y,cvp,'linear',Cvals(i),1); if a>best,best=a;bc=Cvals(i);end,end
out(2)=struct('model',models{2},'cv_accuracy',best,'setting',sprintf('C=%g',bc));
kvals=[1 3 5 7 9 15]; best=-inf; bk=kvals(1);
for i=1:numel(kvals), a=cv_knn(X,Y,cvp,kvals(i)); if a>best,best=a;bk=kvals(i);end,end
out(3)=struct('model',models{3},'cv_accuracy',best,'setting',sprintf('k=%d',bk));
leaves=[1 5 10 20 40]; best=-inf; bl=leaves(1);
for i=1:numel(leaves), a=cv_tree(X,Y,cvp,leaves(i)); if a>best,best=a;bl=leaves(i);end,end
out(4)=struct('model',models{4},'cv_accuracy',best,'setting',sprintf('MinLeafSize=%d',bl));
save('../results/baseline_comparison_dev.mat','out','rep');
fid=fopen('../results/baseline_comparison_dev.csv','w'); fprintf(fid,'Model,CV_Accuracy,Setting\n');
for i=1:4, fprintf(fid,'%s,%.6f,%s\n',out(i).model,out(i).cv_accuracy,out(i).setting); fprintf('%s: %.2f%% | %s\n',out(i).model,100*out(i).cv_accuracy,out(i).setting); end
fclose(fid); fprintf('No official test data were evaluated.\n');
end
function a=cv_svm(X,Y,cvp,kern,C,scale)
p=zeros(size(Y)); for f=1:cvp.NumTestSets, tr=training(cvp,f);va=test(cvp,f);[A,B]=normfold(X,tr,va); args={'KernelFunction',kern,'BoxConstraint',C,'Standardize',false}; if strcmp(kern,'rbf'),args=[args {'KernelScale',scale}];end;m=fitcsvm(A,Y(tr),args{:});p(va)=predict(m,B);end;a=mean(p==Y); end
function a=cv_knn(X,Y,cvp,k)
p=zeros(size(Y));for f=1:cvp.NumTestSets,tr=training(cvp,f);va=test(cvp,f);[A,B]=normfold(X,tr,va);m=fitcknn(A,Y(tr),'NumNeighbors',k,'Standardize',false);p(va)=predict(m,B);end;a=mean(p==Y);end
function a=cv_tree(X,Y,cvp,l)
p=zeros(size(Y));for f=1:cvp.NumTestSets,tr=training(cvp,f);va=test(cvp,f);[A,B]=normfold(X,tr,va);m=fitctree(A,Y(tr),'MinLeafSize',l);p(va)=predict(m,B);end;a=mean(p==Y);end
function [A,B]=normfold(X,tr,va)
mu=mean(X(tr,:),1);s=std(X(tr,:),0,1);s(s==0)=1;A=bsxfun(@rdivide,bsxfun(@minus,X(tr,:),mu),s);B=bsxfun(@rdivide,bsxfun(@minus,X(va,:),mu),s);A(~isfinite(A))=0;B(~isfinite(B))=0;end
function f=dwt18(J,w)
[C,S]=wavedec2(J,2,w);[H1,V1,D1]=detcoef2('all',C,S,1);[H2,V2,D2]=detcoef2('all',C,S,2);cc={H1,V1,D1,H2,V2,D2};f=zeros(1,18);z=1;
for k=1:6,v=cc{k}(:);e=sum(v.^2);den=sum(abs(v));if den==0,ent=0;else,p=abs(v)/den;p(p==0)=[];ent=-sum(p.*log2(p));end;s=std(v);if numel(v)<3||s==0,sk=0;else,sk=(sum((v-mean(v)).^3)/numel(v))/(s^3);end;f(z:z+2)=[e ent sk];z=z+3;end
end
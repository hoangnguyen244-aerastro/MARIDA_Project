%% step3_train_svm.m
% DEVELOPMENT-ONLY split preparation and legacy db4 CV.
% The official test set is mapped but never predicted/evaluated here.
function step3_train_svm()
fprintf('\nSTEP 3: DEVELOPMENT PREPARATION (OFFICIAL TEST SEALED)\n');
load('../data/features_data.mat','all_features','all_labels','valid_files');
tr=read_split('../raw_data/MARIDA/splits/train_X.txt');
va=read_split('../raw_data/MARIDA/splits/val_X.txt');
te=read_split('../raw_data/MARIDA/splits/test_X.txt');
ids=cellfun(@file_id,valid_files,'UniformOutput',false);
a=ismember(ids,tr); b=ismember(ids,va); c=ismember(ids,te);
if any((a+b+c)>1), error('A sample occurs in more than one official split.'); end
dev_idx=find(a|b); test_idx=find(c);
if isempty(dev_idx)||isempty(test_idx), error('Empty official development/test mapping.'); end
X=all_features(dev_idx,:); Y=all_labels(dev_idx);
Cvals=[.1 1 10 100]; scales=[.01 .1 1 10]; best_accuracy=-inf;best_C=1;best_scale=1;
rng(42); cvp=cvpartition(Y,'KFold',5);
for ci=1:numel(Cvals)
 for si=1:numel(scales)
  pred=zeros(size(Y));
  for k=1:cvp.NumTestSets
   q=training(cvp,k); v=test(cvp,k); mu=mean(X(q,:),1); sd=std(X(q,:),0,1);sd(sd==0)=1;
   A=bsxfun(@rdivide,bsxfun(@minus,X(q,:),mu),sd);B=bsxfun(@rdivide,bsxfun(@minus,X(v,:),mu),sd);
   A(~isfinite(A))=0;B(~isfinite(B))=0;
   m=fitcsvm(A,Y(q),'KernelFunction','rbf','BoxConstraint',Cvals(ci),'KernelScale',scales(si),'Standardize',false);
   pred(v)=predict(m,B);
  end
  acc=mean(pred==Y);
  if acc>best_accuracy,best_accuracy=acc;best_C=Cvals(ci);best_scale=scales(si);end
 end
end
mu=mean(X,1);sigma=std(X,0,1);sigma(sigma==0)=1;
A=bsxfun(@rdivide,bsxfun(@minus,X,mu),sigma);A(~isfinite(A))=0;
final_svm=fitcsvm(A,Y,'KernelFunction','rbf','BoxConstraint',best_C,'KernelScale',best_scale,'Standardize',false);
best_gamma=best_scale;train_idx=dev_idx;
save('../models/svm_model.mat','final_svm','mu','sigma','best_C','best_gamma','best_scale','train_idx','test_idx','dev_idx','best_accuracy');
fprintf('Development N=%d; official test mapped N=%d but NOT evaluated.\n',numel(dev_idx),numel(test_idx));
end
function names=read_split(path)
fid=fopen(path,'r');if fid<0,error('Cannot open %s',path);end;C=textscan(fid,'%s');fclose(fid);names=C{1};end
function id=file_id(path)
[~,id,~]=fileparts(path);if length(id)>=3&&strcmp(id(1:3),'S2_'),id=id(4:end);end
end
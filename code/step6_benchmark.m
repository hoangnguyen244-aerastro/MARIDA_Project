%% step6_benchmark.m
% Benchmark the LOCKED deployment model. This measures runtime only; it does
% not compute test performance or alter the locked configuration.
function step6_benchmark()
fprintf('\nSTEP 6: FINAL LOCKED MODEL BENCHMARK\n');
model_file='../models/final_locked_svm.mat';
if ~exist(model_file,'file'), error('Run Step 9 once and keep final_locked_svm.mat.'); end
load(model_file,'final_model','mu','sigma','representation');
load('../data/features_data.mat','valid_files');
load('../models/svm_model.mat','test_idx');
files=valid_files(test_idx); n=numel(files); times=zeros(n,1);
info=dir(model_file); file_size_kb=info.bytes/1024;
for i=1:n
 t=tic; I=double(imread(files{i}));
 if ~strcmp(representation,'nir_842'), error('Final benchmark expects locked NIR-842 representation.'); end
 x=dwt18(I(:,:,8),'db4'); x=sign(x).*log1p(abs(x));
 x=bsxfun(@rdivide,bsxfun(@minus,x,mu),sigma); x(~isfinite(x))=0;
 predict(final_model,x); times(i)=1000*toc(t);
end
benchmark_results=struct('representation',representation,'wavelet','db4', ...
 'model_size_kb',file_size_kb,'avg_inference_ms',mean(times),'std_inference_ms',std(times), ...
 'min_ms',min(times),'max_ms',max(times),'fps',1000/mean(times),'num_images',n);
save('../results/benchmark_results.mat','benchmark_results','times');
fid=fopen('../results/benchmark_results.csv','w');
fprintf(fid,'Representation,Wavelet,Model_Size_KB,Mean_ms,Std_ms,Min_ms,Max_ms,FPS,N_Images\n');
fprintf(fid,'%s,db4,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n',representation,file_size_kb,mean(times),std(times),min(times),max(times),1000/mean(times),n);fclose(fid);
fprintf('NIR benchmark complete: %.2f ms/image, model %.2f KB, N=%d.\n',mean(times),file_size_kb,n);
end
function f=dwt18(J,w)
[C,S]=wavedec2(J,2,w);[H1,V1,D1]=detcoef2('all',C,S,1);[H2,V2,D2]=detcoef2('all',C,S,2);cc={H1,V1,D1,H2,V2,D2};f=zeros(1,18);z=1;
for k=1:6,v=cc{k}(:);e=sum(v.^2);den=sum(abs(v));if den==0,ent=0;else,p=abs(v)/den;p(p==0)=[];ent=-sum(p.*log2(p));end;s=std(v);if numel(v)<3||s==0,sk=0;else,sk=(sum((v-mean(v)).^3)/numel(v))/(s^3);end;f(z:z+2)=[e ent sk];z=z+3;end
end
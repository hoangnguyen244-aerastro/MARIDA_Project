%% generate_manuscript_figures.m
% Publication figures from FROZEN result artifacts only.
% This script does NOT train models, tune parameters, or recompute test predictions.
% Run from code/. MATLAB R2015b compatible.
function generate_manuscript_figures()
outdir='../reports/manuscript_figures';
if ~exist(outdir,'dir'), mkdir(outdir); end
set(0,'DefaultAxesFontName','Arial','DefaultAxesFontSize',9);
set(0,'DefaultTextFontName','Arial','DefaultTextFontSize',9);

%% Fig. 1 Development spectral ablation
T=readtable('../results/spectral_ablation_dev.csv');
names={'Legacy 440/490/560 mix','True RGB gray','Red 665 nm','NIR 842 nm','SWIR1 1600 nm','Red+NIR+SWIR1'};
v=100*T.SVM_CV_Accuracy;
h=figure('Color','w','Units','centimeters','Position',[2 2 16 9]);
bar(v); set(gca,'XTick',1:numel(v),'XTickLabel',names,'XTickLabelRotation',25);
ylabel('5-fold CV accuracy (%)'); ylim([60 80]); grid on; box on;
title('Development-set spectral representation ablation');
savefigs(h,outdir,'fig1_spectral_ablation'); close(h);

%% Fig. 2 Repeated-CV tie-break
T=readtable('../results/spectral_tiebreak_dev.csv');
T=T(1:2,:); m=100*T.Mean_Accuracy; s=100*T.Std_Accuracy;
h=figure('Color','w','Units','centimeters','Position',[2 2 11 8]);
bar(m); hold on; errorbar(1:2,m,s,'.','LineWidth',1.2); hold off;
set(gca,'XTick',1:2,'XTickLabel',{'Red 665 nm','NIR 842 nm'});
ylabel('Repeated 5-fold CV accuracy (%)'); ylim([74 80]); grid on; box on;
title('Development-only representation tie-break (20 repeats)');
savefigs(h,outdir,'fig2_red_nir_tiebreak'); close(h);

%% Fig. 3 Classifier baselines
T=readtable('../results/baseline_comparison_dev.csv');
v=100*T.CV_Accuracy;
h=figure('Color','w','Units','centimeters','Position',[2 2 13 8]);
bar(v); set(gca,'XTick',1:numel(v),'XTickLabel',T.Model);
ylabel('5-fold CV accuracy (%)'); ylim([55 80]); grid on; box on;
title('Classifier comparison on locked NIR-842 representation');
savefigs(h,outdir,'fig3_classifier_baselines'); close(h);

%% Figures 4-5: use saved one-time official-test outputs only
S=load('../results/final_locked_test.mat','TN','FP','FN','TP','fpr','tpr','auc');

%% Fig. 4 Confusion matrix
M=[S.TN S.FP; S.FN S.TP];
h=figure('Color','w','Units','centimeters','Position',[2 2 10 8]);
imagesc(M); axis image; colormap(flipud(gray)); colorbar;
set(gca,'XTick',1:2,'XTickLabel',{'Normal','Anomaly'}, ...
        'YTick',1:2,'YTickLabel',{'Normal','Anomaly'});
xlabel('Predicted class'); ylabel('True class');
title('Official locked test confusion matrix (N = 257)');
mx=max(M(:));
for r=1:2
 for c=1:2
  if M(r,c)>mx/2, tc='w'; else tc='k'; end
  text(c,r,sprintf('%d',M(r,c)),'HorizontalAlignment','center', ...
      'FontWeight','bold','FontSize',11,'Color',tc);
 end
end
savefigs(h,outdir,'fig4_confusion_matrix'); close(h);

%% Fig. 5 ROC
if isempty(S.fpr)||isempty(S.tpr), error('Frozen ROC coordinates are missing. Do not recompute test predictions; inspect final_locked_test.mat.'); end
h=figure('Color','w','Units','centimeters','Position',[2 2 10 8]);
plot(S.fpr,S.tpr,'LineWidth',1.5); hold on; plot([0 1],[0 1],'--','LineWidth',1); hold off;
xlabel('False positive rate'); ylabel('True positive rate'); xlim([0 1]); ylim([0 1]);
axis square; grid on; box on; title(sprintf('Official locked test ROC (AUC = %.3f)',S.auc));
legend({'NIR-842 RBF-SVM','Chance'},'Location','southeast');
savefigs(h,outdir,'fig5_roc_curve'); close(h);

fprintf('\nCreated 5 frozen-result manuscript figures in %s\n',outdir);
fprintf('Each figure is exported as PNG (600 dpi), PDF, and EPS.\n');
fprintf('No model training or test prediction was performed.\n');
end

function savefigs(h,outdir,name)
set(h,'PaperPositionMode','auto');
print(h,fullfile(outdir,[name '.png']),'-dpng','-r600');
print(h,fullfile(outdir,[name '.pdf']),'-dpdf','-painters');
print(h,fullfile(outdir,[name '.eps']),'-depsc','-painters');
end

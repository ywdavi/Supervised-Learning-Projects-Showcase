%% Loading and sampling

load dataset.mat

% plot data
%u=find(labels_tr==1);
%figure(1),hold on
%plot(data_tr(u,1),data_tr(u,2),'r.')
%u=find(labels_tr==2);
%plot(data_tr(u,1),data_tr(u,2),'b.')
%hold off

% stratified sampling
rng('default'); % setting a "seed"
idx_f1=[];
idx_f2=[];
for nclass=1:2
    u=find(labels_tr==nclass);
    idx=randperm(numel(u));
    idx_f1=[idx_f1; u(idx(1:round(numel(idx)/2)))];
    idx_f2=[idx_f2; u(idx(1+round(numel(idx)/2):end))];
end
labels_f1=labels_tr(idx_f1);
labels_f2=labels_tr(idx_f2);
data_f1=data_tr(idx_f1,:);
data_f2=data_tr(idx_f2,:);

%% First Task:
% train five level-1 classifiers on fold1
mdl={};

% SVM with gaussian kernel
rng('default');
mdl{1}=fitcsvm(data_f1, labels_f1, 'KernelFunction', ...
    'gaussian', 'KernelScale', 5);

% SVM with polynomial kernel
rng('default');
mdl{2}=fitcsvm(data_f1, labels_f1, 'KernelFunction', ...
    'polynomial', 'KernelScale', 10);

% decision tree
rng('default');
mdl{3}=fitctree(data_f1, labels_f1, 'SplitCriterion', ...
    'gdi','MaxNumSplits',20);

% Naive Bayes
rng('default');
mdl{4}=fitcnb(data_f1, labels_f1);

% Ensemble of decision trees
rng('default');
mdl{5}=fitcensemble(data_f1, labels_f1);

% obtain the predictions and the scores on fold2 
% (used to train meta-learner)
N=numel(mdl);
Scores=zeros(size(data_f2,1),N);
Predictions=zeros(size(data_f2,1),N);
for ii=1:N
    [predictions, scores]=predict(mdl{ii},data_f2);
    Scores(:,ii)=scores(:,1);
    Predictions(:,ii)=predictions;
end

% train the stacked classifier on fold2 
% (both on scores and predictions)

rng('default');
stackedModel_onScores = fitcensemble(Scores, labels_f2, ...
    'Method','Bag',...
    'NumLearningCycles',200); %,'OptimizeHyperparameters','all');  actually chooses the parameter by himself, not used
stackedModel_onPred = fitcensemble(Predictions, labels_f2, ...
    'Method', 'Bag',...
    'NumLearningCycles',200); %,'OptimizeHyperparameters','all'); 

mdl{N+1}=stackedModel_onScores;
mdl{N+2}=stackedModel_onPred;

% alternatives: 'AdaBoostM1', 'GentleBoost', 'LogitBoost',...

ACC=[];
Predictions=zeros(size(data_te,1),N);
Scores=zeros(size(data_te,1),N);

for ii=1:N
    [predictions, scores]=predict(mdl{ii},data_te);
    Predictions(:,ii)=predictions;
    Scores(:,ii)=scores(:,1);
    ACC(ii)=numel(find(predictions==labels_te))/numel(labels_te);
end

% predictions of the stacked classifiers (score-based and 
% prediction-based) over the test set

predictions_scores = predict(mdl{N+1}, Scores);
prediction=predict(mdl{N+2}, Predictions);

ACC(N+1)=numel(find(predictions_scores==labels_te))/numel(labels_te);
ACC(N+2)=numel(find(prediction==labels_te))/numel(labels_te);

ACC
%% (First Task with AdaBoost):
% not reported

mdl={};
rng('default');
mdl{1}=fitcsvm(data_f1, labels_f1, 'KernelFunction', ...
    'gaussian', 'KernelScale', 5);
rng('default');
mdl{2}=fitcsvm(data_f1, labels_f1, 'KernelFunction', ...
    'polynomial', 'KernelScale', 10);
rng('default');
mdl{3}=fitctree(data_f1, labels_f1, 'SplitCriterion', ...
    'gdi','MaxNumSplits',20);
rng('default');
mdl{4}=fitcnb(data_f1, labels_f1);
rng('default');
mdl{5}=fitcensemble(data_f1, labels_f1);

N=numel(mdl);
Scores=zeros(size(data_f2,1),N);
Predictions=zeros(size(data_f2,1),N);
for ii=1:N
    [predictions, scores]=predict(mdl{ii},data_f2);
    Scores(:,ii)=scores(:,1);
    Predictions(:,ii)=predictions;
end

rng('default');
stackedModel_onScores = fitcensemble(Scores, labels_f2, ...
    'Method','AdaBoostM1',...
    'Learners','tree');
stackedModel_onPred = fitcensemble(Predictions, labels_f2, ...
    'Method', 'AdaBoostM1',...
    'Learners','tree'); 

mdl{N+1}=stackedModel_onScores;
mdl{N+2}=stackedModel_onPred;

ACC2=[];
Predictions=zeros(size(data_te,1),N);
Scores=zeros(size(data_te,1),N);

for ii=1:N
    [predictions, scores]=predict(mdl{ii},data_te);
    Predictions(:,ii)=predictions;
    Scores(:,ii)=scores(:,1);
    ACC2(ii)=numel(find(predictions==labels_te))/numel(labels_te);
end

predictions_scores = predict(mdl{N+1}, Scores);
prediction=predict(mdl{N+2}, Predictions);

ACC2(N+1)=numel(find(predictions_scores==labels_te))/numel(labels_te);
ACC2(N+2)=numel(find(prediction==labels_te))/numel(labels_te);

ACC2
%% Second Task:
% five level-1 classifiers are trained on the whole training set

mdl={};

% SVM with gaussian kernel 
rng('default')
mdl{1}=fitcsvm(data_tr, labels_tr, 'KernelFunction','gaussian', ...
        'KernelScale', 5);

% SVM with polynomial kernel 
mdl{2}=fitcsvm(data_tr, labels_tr, 'KernelFunction','polynomial', ...
        'KernelScale', 10);

% Decision Tree 
rng('default')
mdl{3}=fitctree(data_tr, labels_tr, 'SplitCriterion','gdi', ...
        'MaxNumSplits', 20);

% Naive Bayes 
rng('default')
mdl{4}=fitcnb(data_tr, labels_tr); 

% Ensemble of decision trees
rng('default')
mdl{5}=fitcensemble(data_tr, labels_tr); 

N=numel(mdl);
Scores=zeros(size(data_tr,1),N);
Predictions=zeros(size(data_tr,1),N);
for ii=1:N
    [predictions, scores]=predict(mdl{ii},data_tr);
    Scores(:,ii)=scores(:,1);
    Predictions(:,ii)=predictions;
end
    
% the two stacked classifiers are built on the whole training set
rng('default');
stackedModel_onScores = fitcensemble(Scores, labels_tr, ...
    'Method','Bag');
stackedModel_onPred = fitcensemble(Predictions, labels_tr, ...
    'Method', 'Bag'); 

mdl{N+1}=stackedModel_onScores;
mdl{N+2}=stackedModel_onPred;

% checking the performances on the test set
ACC3=[];
Predictions=zeros(size(data_te,1),N);
Scores=zeros(size(data_te,1),N);

for ii=1:N
    [predictions, scores]=predict(mdl{ii},data_te);
    Predictions(:,ii)=predictions;
    Scores(:,ii)=scores(:,1);
    ACC3(ii)=numel(find(predictions==labels_te))/numel(labels_te);
end

predictions_scores = predict(mdl{N+1}, Scores);
prediction=predict(mdl{N+2}, Predictions);

ACC3(N+1)=numel(find(predictions_scores==labels_te))/numel(labels_te);
ACC3(N+2)=numel(find(prediction==labels_te))/numel(labels_te);

ACC3
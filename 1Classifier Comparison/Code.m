% Initialize final accuracy matrix

accuracy_matrix = [];

for ndataset = 1:4
    switch ndataset
        case 1
            load dataset1.mat
        case 2
            load dataset2.mat
        case 3
            load dataset3.mat
        case 4
            load dataset4.mat
        otherwise
    end
    
    accuracy_TREE = [];
    accuracy_QDA = [];
    accuracy_SVM_LIN = [];
    accuracy_SVM_RBF = [];
    accuracy_KNN = [];
    
    % crossvalidation

    for ntimes = 1:5
        % train and test subsets
        idx_tr = []; % indexes used for training
        idx_te = []; % indexes used for testing
        for nclass = 1:2
            u = find(labels == nclass);
            idx = randperm(numel(u));
            idx_tr = [idx_tr; u(idx(1:round(numel(idx) / 2)))];
            idx_te = [idx_te; u(idx(1 + round(numel(idx) / 2):end))];
        end
    
        labels_tr = labels(idx_tr);
        labels_te = labels(idx_te);
        data_tr = data(idx_tr, :);
        data_te = data(idx_te, :);

        % training classifiers
        SVM_LIN = fitcsvm(data_tr, labels_tr, 'KernelFunction', 'linear', 'KernelScale', 1);
        SVM_RBF = fitcsvm(data_tr, labels_tr, 'KernelFunction', 'gaussian', 'KernelScale', sqrt(2));
        KNN = fitcknn(data_tr, labels_tr, 'Distance', 'Euclidean', 'NumNeighbors', 17); %sqrt(300)
        TREE = fitctree(data_tr, labels_tr, 'SplitCriterion', 'gdi', 'MaxNumSplits', 10);
        QDA = fitcdiscr(data_tr, labels_tr, 'DiscrimType', 'Quadratic');
        
        % prediction and accuracy evaluation
        prediction_SVM_LIN = predict(SVM_LIN, data_te);
        prediction_SVM_RBF = predict(SVM_RBF, data_te);
        prediction_KNN = predict(KNN, data_te);
        prediction_TREE = predict(TREE, data_te);
        prediction_QDA = predict(QDA, data_te);

        accuracy_SVM_LIN1 = numel(find(prediction_SVM_LIN == labels_te)) / numel(labels_te);
        accuracy_SVM_RBF1 = numel(find(prediction_SVM_RBF == labels_te)) / numel(labels_te);
        accuracy_KNN1 = numel(find(prediction_KNN == labels_te)) / numel(labels_te);
        accuracy_TREE1 = numel(find(prediction_TREE == labels_te)) / numel(labels_te);
        accuracy_QDA1 = numel(find(prediction_QDA == labels_te)) / numel(labels_te);

        % switch train and test
        SVM_LIN2 = fitcsvm(data_te, labels_te, 'KernelFunction', 'linear', 'KernelScale', 1);
        SVM_RBF2 = fitcsvm(data_te, labels_te, 'KernelFunction', 'gaussian', 'KernelScale', sqrt(2));
        KNN2 = fitcknn(data_te, labels_te, 'Distance', 'Euclidean', 'NumNeighbors', 17);
        TREE2 = fitctree(data_te, labels_te, 'SplitCriterion', 'gdi', 'MaxNumSplits', 10);
        QDA2 = fitcdiscr(data_te, labels_te, 'DiscrimType', 'Quadratic');
        
        % prediction and accuracy evaluation
        prediction_SVM_LIN2 = predict(SVM_LIN2, data_tr);
        prediction_SVM_RBF2 = predict(SVM_RBF2, data_tr);
        prediction_KNN2 = predict(KNN2, data_tr);
        prediction_TREE2 = predict(TREE2, data_tr);
        prediction_QDA2 = predict(QDA2, data_tr);

        accuracy_SVM_LIN2 = numel(find(prediction_SVM_LIN2 == labels_tr)) / numel(labels_tr);
        accuracy_SVM_RBF2 = numel(find(prediction_SVM_RBF2 == labels_tr)) / numel(labels_tr);
        accuracy_KNN2 = numel(find(prediction_KNN2 == labels_tr)) / numel(labels_tr);
        accuracy_TREE2 = numel(find(prediction_TREE2 == labels_tr)) / numel(labels_tr);
        accuracy_QDA2 = numel(find(prediction_QDA2 == labels_tr)) / numel(labels_tr);

        % average between the two accuracy
        accuracy_SVM_LIN(ntimes) = (accuracy_SVM_LIN1 + accuracy_SVM_LIN2) / 2;
        accuracy_SVM_RBF(ntimes) = (accuracy_SVM_RBF1 + accuracy_SVM_RBF2) / 2;
        accuracy_KNN(ntimes) = (accuracy_KNN1 + accuracy_KNN2) / 2;
        accuracy_TREE(ntimes) = (accuracy_TREE1 + accuracy_TREE2) / 2;
        accuracy_QDA(ntimes) = (accuracy_QDA1 + accuracy_QDA2) / 2;

    end

    % Storing cross validated accuracy

    accuracy_matrix(ndataset,:) = [mean(accuracy_TREE), ...
        mean(accuracy_QDA), mean(accuracy_SVM_LIN), ...
        mean(accuracy_SVM_RBF), mean(accuracy_KNN)];
end

% Converting the matrix into ranks

ranks = [];

for row = 1:4
    row_values = accuracy_matrix(row,:);
    [~, sorted_indices] = sort(row_values, 'descend');
    ranks(row, sorted_indices) = 1:5;
end

% Averaging the ranks

mean_rank = mean(ranks, 1);

% Computing the critical difference 
% With 5 classifiers:
k = 5;
q_005 = 2.728; %2.569; if 4
q_010 = 2.459; %2.291;

% With 4 datasets:
N = 4;

CD = q_010 * sqrt((k * (k + 1)) / (6 * N));



% Plots:

%- scatterplot

for ndataset = 1:4
    switch ndataset
        case 1
            load dataset1.mat
        case 2
            load dataset2.mat
        case 3
            load dataset3.mat
        case 4
            load dataset4.mat
        otherwise
    end

    % Extracting columns
    x = data(:, 1);
    y = data(:, 2);

    % Define colors based on labels
    unique_labels = unique(labels);
    colors = lines(length(unique_labels));

    % Plotting the scatter plot
    subplot(2, 2, ndataset); % Create a subplot in a 2x2 grid
    hold on;
    for i = 1:length(unique_labels)
        idx = labels == unique_labels(i);
        scatter(x(idx), y(idx), [], colors(i, :), 'filled');
    end

    xlabel('X-axis Label');
    ylabel('Y-axis Label');
    title(['Dataset ', num2str(ndataset)]);
    legend('Label 1', 'Label 2');
    hold off;
end

%- critical difference diagram

figure;
hold on;
for i = 1:5
    x_start = mean_rank(i) - CD / 2;
    x_end = mean_rank(i) + CD / 2;
  
    plot([x_start, x_end], [i, i], 'LineWidth', 2, 'Color', 'b'); 
    plot(mean_rank(i), i, 'x', 'MarkerSize', 8, 'Color', 'k');
end

% Customize plot
xlim([1, 5])
xticks(1:5)
ylim([0.5, 5 + 0.5]); %
yticks(1:5);          %
yticklabels({'Decision Tree Classifier', 'Quadratic Discriminant Analysis', ...
    'Linear SVM', 'Medium Gaussian SVM', "K-Nearest Neighbors"});
xlabel('Average Rank');
title('Critical Difference Diagram');
legend(['CD = ', num2str(CD)]);
grid on;
hold off;




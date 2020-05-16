% CLASSIFICATION
% Goal: Distinguish setosa from non-setosa

clear
clc
close all

%% Initialization

load iris_dataset.mat;
x = zscore(irisInputs([1 2],:)'); % take as input normalized sepal length 
                                  % and width
t = irisTargets(1,:)'; % target: is setosa or not
gplotmatrix(x,[],t);

%% Perceptron (automatic)

% train perceptron with stochastic gradient descent
% loss takes in account of the distance of misclassified (when all points 
% are correctly classified, it stops improving!) points from the decision 
% boundary
net = perceptron;
net = train(net, x', t');

% plot the result
gscatter(x(:,1),x(:,2),t); % show the points
hold on;
s = -2:0.01:2.5; % sampling intervals
% draw the separating hyperplane by solving sgn(w_0 + x_1 w_1 + x_2 w_2)=0
plot(s, -(s * net.IW{1}(1) + net.b{1} ) / net.IW{1}(2),'k');
axis manual

% evaluate the result with the confusion matrix

%% Perceptron (by hand)

% prepare the input
n_samples = size(x,1);
perc_t = t;
perc_t(perc_t == 0) = -1; % target values must be 1 or -1
ind = randperm(n_samples); % generate shuffling vector
perc_t = perc_t(ind); % reorder target
x_perc = x(ind,:); % reorder input

% perceptron algorithm
w = ones(1,3); % initialize the weights
for jj = 1:10
    for ii = 1:n_samples
        if sign(w * [1 x_perc(ii,:)]') ~= perc_t(ii) % if misclassified
            w = w + [1 x_perc(ii,:)] * perc_t(ii); % update weight
        end
        if mod(ii,50) == 0
            plot(s, -(s * w(2) + w(1) ) / w(3), 'r');
        end
    end
end

% plot and compare to the automated procedure
figure();
gscatter(x_perc(:,1),x_perc(:,2),perc_t);
hold on;
axis manual;
s = -2:0.01:2.5;
h(1) = plot(s, -(s * net.IW{1}(1) + net.b{1} ) / net.IW{1}(2),'k');
h(2) = plot(s, -(s * w(2) + w(1) ) / w(3),'r');
legend(h,{'Perceptron (func)' 'Perceptron (hand)'})

%% Logistic regression
t = t+1;
[B, ~, stats] = mnrfit(x,t);

pihat = mnrval(B,x);
[~, t_pred] = max(pihat,[],2);
confusionmat(t,t_pred)

h(3) = plot(s, -(s * B(2) + B(1) ) / B(3),'b');
legend(h,{'Perceptron (func)' 'Perceptron (hand)' 'Logistic'})

%% Multinomial logistic regression

[t, ~] = find(irisTargets ~= 0);
[B_mul, dev_mul, stats_mul] = mnrfit(x,t);

pihat = mnrval(B_mul,x);
[~, t_pred] = max(pihat,[],2);
confusionmat(t,t_pred)

figure();
gscatter(x(:,1),x(:,2),t);
hold on;
axis manual

[a, b] = meshgrid(-3:0.1:3,-3:0.1:4);
axis tight
pihat = mnrval(B_mul,[a(:),b(:)]);
[~, pred] = max(pihat,[],2);
gscatter(a(:),b(:),pred);

%% Naive bayes
nb_model = fitcnb(x,t);

t_pred = predict(nb_model,x);
confusionmat(t,t_pred)

figure();
gscatter(x(:,1),x(:,2),t);
hold on;
axis manual

[a, b] = meshgrid(-3:0.1:3,-3:0.1:4);
axis tight;
pred = predict(nb_model,[a(:),b(:)]);
gscatter(a(:),b(:),pred);


%% Generative abilities of NB
param = nb_model.DistributionParameters;
prior = cumsum(nb_model.Prior);
n_dim = size(param,2);

n_gen = 1000;
gendata = zeros(n_gen,n_dim);
gentarget = zeros(n_gen,1);

for ii = 1:n_gen
    gentarget(ii) = find(prior > rand(),1);
    for jj = 1:n_dim
        mu = param{gentarget(ii),jj}(1);
        sigma = param{gentarget(ii),jj}(2);
        gendata(ii,jj) = normrnd(mu,sigma);
    end
end

figure();
gscatter(gendata(:,1),gendata(:,2),gentarget);

%% KNN classifier
knn_model = fitcknn(x, t, 'NumNeighbors', 2);
t_pred = predict(knn_model,x);
confusionmat(t, t_pred)

figure();
gscatter(x(:,1),x(:,2),t);
hold on;
axis manual

[a, b] = meshgrid(-3:0.1:3,-3:0.1:4);
axis tight
pred = predict(knn_model,[a(:),b(:)]);
gscatter(a(:),b(:),pred);
title('K-NN classifier');

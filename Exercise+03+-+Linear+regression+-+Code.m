% REGRESSION
% LINEAR REGRESSION
% Goal: predict the petal width of setosa by using petal length

clear;
clc;
close all;

%% Initialization
load iris_dataset.mat; % has irisInputs (4x150: sepal len, sepal wid, 
                       % petal len, petal wid) and irisTargets (3x150 
                       % setosa, virginica, versicolor)

% visualize data
%{
 In the matrix of scatter plots, the x-axis of the leftmost column of 
scatter plots corresponds to sepal length, the first column in meas. 
Similarly, the y-axis of the bottom row of scatter plots corresponds to 
petal width, the last column in meas. Therefore, the scatter plot in the 
bottom left of the matrix compares sepal length values (along the x-axis) 
to petal width values (along the y-axis). The color of each point depends 
on the species of the flower.

The diagonal plots are histograms rather than scatter plots. For example, 
the plot in the top left of the matrix shows the distribution of sepal 
length values for each species of flower.
%}
figure();
gplotmatrix(irisInputs', [], irisTargets');
title('Inputs');

% visualize petal len and wid
x = irisInputs(3,:)';
t = irisInputs(4,:)';

figure();
plot(x, t, 'bo');
title('Petal length and width')

%% Preprocessing
x = zscore(x); 
t = zscore(t);
%{ 
  the mean now is 0, standard deviation 1
  Rather than showing a data point on its own scale, z-scores show how 
  many standard deviations a data point is from the mean (average).
%} 
figure();
plot(x, t, 'bo');
title('Standardized data');

%% Fit
%{
  Example fittype: ft3 = fittype({'cos(x)','1'},'coefficients',{'a1','a2'})
  ft3 = 
     Linear model:
     ft3(a1,a2,x) = a1*cos(x) + a2
%}
fit_specifications = fittype( {'1', 'x'}, 'independent', 'x',...
                          'dependent', 't', 'coefficients', {'w0', 'w1'} );
%{
  Fit returns:
  fitresult: the linear model
  gof: statistics
%}
[fitresult, gof] = fit( x, t, fit_specifications);

hold on;
plot(fitresult);
title("Fit result");

%% Other fit functions
%{
Example and explaination of fitlm

ans=4×4 table
                    Estimate        SE          tStat        pValue  
                   __________    _________    _________    __________

    (Intercept)        47.977       3.8785        12.37    4.8957e-21
    x1             -0.0065416    0.0011274      -5.8023    9.8742e-08
    x2              -0.042943     0.024313      -1.7663       0.08078
    x3              -0.011583      0.19333    -0.059913       0.95236


The Coefficient property includes these columns:

Estimate — Coefficient estimates for each corresponding term in the model. 
For example, the estimate for the constant term (intercept) is 47.977.

SE — Standard error of the coefficients.

tStat — t-statistic for each coefficient to test the null hypothesis that 
the corresponding coefficient is zero against the alternative that it is 
different from zero, given the other predictors in the model. Note that 
tStat = Estimate/SE. For example, the t-statistic for the intercept is 
47.977/3.8785 = 12.37.

pValue — p-value for the t-statistic of the hypothesis test that the 
corresponding coefficient is equal to zero or not. For example, the p-value
 of the t-statistic for x2 is greater than 0.05, so this term is not 
significant at the 5% significance level given the other terms in the model.


The summary statistics of the model are:

Number of observations — Number of rows without any NaN values. For example, 
Number of observations is 93 because the MPG data vector has six NaN values 
and the Horsepower data vector has one NaN value for a different observation, 
where the number of rows in X and MPG is 100.

Error degrees of freedom — n – p, where n is the number of observations, 
and p is the number of coefficients in the model, including the intercept. 
For example, the model has four predictors, so the Error degrees of freedom 
is 93 – 4 = 89.

Root mean squared error — Square root of the mean squared error, which 
estimates the standard deviation of the error distribution.

R-squared and Adjusted R-squared — Coefficient of determination and 
adjusted coefficient of determination, respectively. For example, the 
R-squared value suggests that the model explains approximately 75% of the 
variability in the response variable MPG.

F-statistic vs. constant model — Test statistic for the F-test on the 
regression model, which tests whether the model fits significantly better 
than a degenerate model consisting of only a constant term.

p-value — p-value for the F-test on the model. For example, the model is 
significant with a p-value of 7.3816e-27.
%}
ls_model = fitlm(x, t);

%% By hand regression
n_sample = length(x);
Phi = [ones(n_sample,1) x];
mpinv = pinv(Phi' * Phi) * Phi';
w = mpinv * t;

hat_t = Phi * w;
bar_t = mean(t);
SSR = sum((t-hat_t).^2);
R_squared = 1 - SSR / sum((t-bar_t).^2);

%% Ridge and lasso
lambda = 10^(-10);
ridge_coeff = ridge(t, Phi, lambda);

[lasso_coeff, lasso_fit] = lasso(Phi, t);



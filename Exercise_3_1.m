% LINEAR REGRESSION
 
%{
  Goal: 
predict acceleration of a car given all the other features
on carsmall:
- visually select the most appropriate variables by looking at the
scatterplot
- prepare them for a linear regression
%}

%% Prior analysis
% Load data
load carsmall.mat;

% Inspect data and select interesting data
to_consider_features = [Acceleration Cylinders Displacement Horsepower Model_Year Weight];
gplotmatrix(to_consider_features);
% From the scatterplot we get that year is not useful for acceleration,
% because all years map to almost the same value
% Weight does not seem to be related to acceleration
% For the others, there is a visible relation

%% Preprocessing
%{
 Notice: by inspection on Horsepower we see that there's a NaN value
  Those can be identified by means of isnan()
%}
% take the useful data, clean it
x = to_consider_features(:, [2 3 4]);
%{ 
repmat(A, 2, 3) replicates matrix A in a "matrix" 2x3
[AAA
AAA]
nanmean gets the mean discarding nan
%}
% standardize features by hand, to deal with NaN
x = ((x - repmat(nanmean(x), 100, 1)) ./ repmat(nanstd(x), 100, 1));

% target
t = to_consider_features(:,1);
t = zscore(t);

% Show standardized data
figure();
subplot(1, 3, 1);
plot(x(:, 1), t, 'bo');
title("Cylinders");

subplot(1, 3, 2);
plot(x(:, 2), t, 'bo');
title("Displacement");

subplot(1, 3, 3);
plot(x(:, 3), t, 'bo');
title("Horsepower");

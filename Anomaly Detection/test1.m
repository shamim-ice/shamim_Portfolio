clear ; close all; clc

fprintf('Visualizing example dataset for outlier detection.\n\n');

%  The following command loads the dataset. You should now have the
%  variables X, Xval, yval in your environment
datacell = textscan( regexprep( fileread('df.csv'), '\$', '0' ), '%f%f', 'delimiter', ',', 'HeaderLines', 1);

snr = datacell{1};
energy = datacell{2};
%X = table(snr, energy);
X=[snr,energy];

%  Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bx');
xlabel('snr_{dB}');
ylabel('Energy level');


fprintf('Program paused. Press enter to continue.\n');
pause


%% ================== Part 2: Estimate the dataset statistics ===================

fprintf('Visualizing Gaussian fit.\n\n');

%  Estimate mu and sigma2
[mu, sigma2] = estimateGaussian(X);

%  Returns the density of the multivariate normal at each data point (row) 
%  of X
p = multivariateGaussian(X, mu, sigma2);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('snr_{dB}');
ylabel('Energy level');

fprintf('Program paused. Press enter to continue.\n');
pause;


pval = multivariateGaussian(X, mu, sigma2);

[epsilon, F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n');
fprintf('   (you should see a Best F1 value of  0.875000)\n\n');

%  Find the outliers in the training set and plot the
outliers = find(p < epsilon);

%  Draw a red circle around those outliers
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Part 4: Multidimensional Outliers ===================
%  We will now use the code from the previous part and apply it to a 
%  harder problem in which more features describe each datapoint and only 
%  some features indicate whether a point is an outlier.
%

%  Loads the second dataset. You should now have the
%  variables X, Xval, yval in your environment
load('ex8data2.mat');

%  Apply the same steps to the larger dataset
[mu, sigma2] = estimateGaussian(X);

%  Training set 
p = multivariateGaussian(X, mu, sigma2);

%  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2);

%  Find the best threshold
[epsilon, F1] = selectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 1.38e-18)\n');
fprintf('   (you should see a Best F1 value of 0.615385)\n');
fprintf('# Outliers found: %d\n\n', sum(p < epsilon));


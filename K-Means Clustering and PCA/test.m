clear ; 
close all; 
clc;

fprintf('Finding closest centroids.\n\n');

% Load an example dataset that we will be using
%load('ex7data2.mat');
datacell = textscan( regexprep( fileread('df3.csv'), '\$', '0' ), '%f%f', 'delimiter', ',', 'HeaderLines', 1);

snr = datacell{1};
energy = datacell{2};
%X = table(snr, energy);
X=[snr,energy];
%load('data.mat');
%scatter(energySum(:,1),energySum(:,2),'b');
scatter(X(:,1), X(:,2));
xlabel('snr_dB');
ylabel('Energy level');
% Select an initial set of centroids
%K = 3; % 3 Centroids
K=3;
%initial_centroids = [3 3; 6 2; 8 5];
initial_centroids = kMeansInitCentroids(X, K);

% Find the closest centroids for the examples using the
% initial_centroids
idx = findClosestCentroids(X, initial_centroids);

fprintf('Closest centroids for the examples: \n')
fprintf(' %d', idx(1:20));
%fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ===================== Part 2: Compute Means =========================

%
%fprintf('\nComputing centroids means.\n\n');

%  Compute means based on the closest centroids found in the previous part.
%centroids = computeCentroids(X, idx, K);

%fprintf('Centroids computed after initial finding of closest centroids: \n')
%fprintf(' %f %f \n' , centroids');
%scatter(centroids(:,1),centroids(:,2),'r');


max_iters = 10;



% Run K-Means algorithm. The 'true' at the end tells our function to plot
% the progress of K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf(' %f %f \n' , centroids');

fprintf('Closest centroids for the examples: \n')
fprintf(' %d', idx(1:20));
fprintf('\nK-Means Done.\n\n');


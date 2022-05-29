function covA = ourCov(A)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Aim: This function calulates the covariance matrix of the matrix A
% Input: 
% -A- Is the an n x d matrix as input where n denotes the number of 
% points and d their dimensionality
% Output: 
% -covA- computed covariance matrix 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code is inspried by https://datascienceplus.com/understanding-the-covariance-matrix/
% Determine the dimension of the array. Or in other words: How many
% observations of the components are in the data set. 
n = size(A,1); 
% calculate the mean of each column. Or in other words: Whats the mean of
% the variable over all observations. 
mu = (ones(1,n) * A) / n;
% Performing the well known formula for computing the covariance matrix
% 1. Step: substract the mean of the initial matrix
A_mean_subtract = A - mu(ones(1,n), :);
% 2. Step: Put it together.
covA = (A_mean_subtract.' * A_mean_subtract) / (n - 1);
% Covariance matrix is symmetric, so you just need to compute one half of 
% it (and copy the rest) and has variance of xi at main diagonal.
end 
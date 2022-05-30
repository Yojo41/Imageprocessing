function [dataMean, sort_EigVec, sort_EigVal] = our_pca(origData,dim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%origData: original data values (Nx2 matrix)
%recData: reconstructed data
%dataMean: mean (center) of data values (1x2 matrix)
%V: 2x2 matrix containing eigenvectors in colums (sorted by descending 
% eigenvalues)
%D: 1x2 vector containing eigenvalus (sorted descending)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------------------------------------------
%Aim: eigenvalues of the covariance matrix encode the variability of the 
%data in an orthogonal basis that captures as much of the data's 
%variability as possible in the first few basis functions (aka the 
%principle component basis)
%--------------------------------------------------------------------------
origData = transpose(origData); 

%if count > dim 
%    error('count <= dim')
%end 
%1.Step: Calculate empirical mean (center) and substract it from the data 
dataMean = mean(origData,1)'; 
origData = (origData' - dataMean)'; 
%2.Step: Calculate eigenvalues and eigenvectors of covariance matrix.
%V = eigenvectors, D= diagonalmatrix of eigenvalues --> A*V = V*D
[EigVec,EigVal] = eig(ourCov(origData));  
%3.Step: Sort eigenvalues and eigenvectors in descending order. 
%Retain only the k eigenvectors with highest corresponding eigenvalues
[sort_B, sort_I ]= sort(diag(EigVal), 'descend'); % == latent 
sort_EigVal = sort_B; 
for j = 1:dim
sort_EigVec(:,j) = EigVec(:,sort_I(j)); 
end 
%sort_EigVec = -sort_EigVec; 
%--------------------------------------------------------------------------
end 


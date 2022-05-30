%% run.m

function run(ourData,split)
% Input: ourData 
% split: 
%calculation of the dimensions of the inputdata 
dim = min(size(ourData)); 
len = length(ourData); 
%% Exercise 1 
covmatrix = ourCov(ourData'); 
%test_cov = cov(ourData'); 
%isalmost(covmatrix,test_cov,1e-10)
if dim == 2
    plot_ourCov(ourData'); 
    title('covariance matrix')
end 
%% Exercise 2 (a)
[dataMean, EigVec, EigVal] = our_pca(ourData,dim); 
%[coeff,score,latent,tsquared,explained,mu] = pca(ourData','NumComponents',2); 
%isalmost(abs(coeff),abs(EigVec),1e-10); 
%% Exercise 2 (b) 
%Multiply the original data by the principal component vectors to get the 
%projections of the original data on the principal component vector space.
%scores will be orthogonal by construction, which you can check corr(scores)
scores = (ourData - dataMean)' *  EigVec; 
%isalmost(abs(scores),abs(score),1e-10); 

recData = ((scores * EigVec')' + repmat(dataMean,1,len))';  
%reconstructed = score * coeff' + mu;
%isalmost(abs(recData),abs(reconstructed),1e-10); 

plot2DPCA(ourData',dataMean',recData,EigVec,EigVal',1,1)
title('PCA 2D - original and reconstructed data')
%% Exercise 3 
if split == 1 
    % going through the 1st,2nd,3rd.... pca component
    for i = 1:dim
scores_ms = (ourData - dataMean)' *  EigVec(:,i); 
recData_ms = ((scores_ms * EigVec(:,i)')' + repmat(dataMean,1,len))'; 
plot2DPCA(ourData',dataMean',recData_ms,EigVec,EigVal',1,1)
title('main and side components of the PCA are used for reconstruction')
    end 
end 
%% Exercise 4 (b) 
if split == 2   
plot3DPCA(ourData',dataMean',EigVec,EigVal',1,1)
end 
end 

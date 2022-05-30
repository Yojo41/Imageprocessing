%% run.m

function run(ourData)
% rec = 1 for main vector  
% rec = 2 for side vector 
covmatrix = ourCov(ourData'); 
test_cov = cov(ourData'); 

dim = min(size(ourData)); 

[dataMean, EigVec, EigVal] = our_pca(ourData,dim); 
[coeff,score,latent] = pca(ourData'); 
% Multiply the original data by the principal component vectors to get the 
% projections of the original data on the principal component vector space.
scores = (ourData - dataMean)' *  EigVec; 
% scores will be orthogonal by construction, which you can check
% corr(scores)

recData = ((scores * EigVec')' + repmat(dataMean,1,50))';  
%reconstructed = score * coeff' + repmat(dataMean, 1, 50)';

if dim == 2
plot2DPCA(ourData',dataMean',recData,EigVec,EigVal',1,1)
elseif dim == 3 
plot3DPCA(origData',dataMean',recData,EigVec,EigVal',1,1)
end 
end 

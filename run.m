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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Exercise 5

% close all;
% clear all;
% clc;

%% a) Generate shape
load('shapes.mat');

size_of_data_ex5 = size(aligned);
nPoints_ex5  = size_of_data_ex5(1);
nDimensions_ex5  = size_of_data_ex5(2);
nShapes_ex5  = size_of_data_ex5(3);

% Compute PCA()
reshaped_data_ex5 = reshape(aligned, nPoints_ex5 * nDimensions_ex5, nShapes_ex5);
[mean_shapes_ex5, eig_vectors_ex5, eig_values_ex5] = our_pca(reshaped_data_ex5, nPoints_ex5 * nDimensions_ex5);


figure;
for i = 1:nShapes_ex5
   plot(aligned(:, 1, i), aligned(:, 2, i), color='#4DBEEE', LineStyle=':');
   hold on
end
plot(mean_shapes_ex5(1:nPoints_ex5), mean_shapes_ex5((nPoints_ex5+1):(nPoints_ex5*nDimensions_ex5)), color='red');
title('Training shapes vs. mean shape');


%% b) Experiments


%% Experiment 1.a - Using all eigenvectors
b_ex5  = ones(nPoints_ex5  * nDimensions_ex5 , 1);
figure;
plotShape(eig_vectors_ex5, b_ex5, mean_shapes_ex5, 'blue');
title('Shape generation with all eigenvectors');
legend('Constructed shape', 'Mean shape', Location='southeast');

%% Experiment 1.b - Using 0 eigenvectors
b_ex5  = zeros(nPoints_ex5  * nDimensions_ex5 , 1);
figure;
plotShape(eig_vectors_ex5, b_ex5, mean_shapes_ex5, 'blue');
title('Shape generation - 0 eigenvectors');
legend('Constructed shape', 'Mean shape', Location='southeast');

%% Experiment 2 - Using only the first eigenvector (accounts for 52.42% of the data)
b_ex5  = zeros(nPoints_ex5  * nDimensions_ex5 , 1);
b_ex5(1) = 1;
figure;
plotShape(eig_vectors_ex5, b_ex5, mean_shapes_ex5, 'blue');
title('Shape generation - first eigenvector (52.42%)');
legend('Constructed shape', 'Mean shape', Location='southeast');



%% Experiment 3.a) - Using only the first eigenvector with varying coefficients ( b_1 in [10, 20, 30, .., 300] )
b_ex5  = zeros(nPoints_ex5  * nDimensions_ex5 , 1);
b_ex5(1) = 1;
figure;


length_ex5 = 30;
blue_ex5 = [0, 0, 1];
%blueish_ex5 = [0.3010 0.7450 0.9330];
blueish_ex5 = [193 245 237]/255;
colors_p_ex5 = [linspace(blue_ex5(1),blueish_ex5(1),length_ex5)', linspace(blue_ex5(2),blueish_ex5(2),length_ex5)',linspace(blue_ex5(3),blueish_ex5(3),length_ex5)'];


b_aux_ex5 = b_ex5;
for i = 1:length_ex5
    b_aux_ex5(1) = b_ex5(1) + 10*i;
    color_ex5 = colors_p_ex5(i, :);
    plotShape(eig_vectors_ex5, b_aux_ex5, mean_shapes_ex5, color_ex5);
    hold on
end
title('Shape generation - first eigenvector with varying coefficients')
%legend('Constructed shape', 'Mean shape', Location='southeast');


%% Experiment 3.b) - Using only the first 2 eigenvectors with varying coefficients ( b_1 in [10, 20, 30, .., 300] )
b_ex5  = zeros(nPoints_ex5  * nDimensions_ex5 , 1);
b_ex5(1:2) = 1;
figure;


length_ex5 = 30;
blue_ex5 = [0, 0, 1];
%blueish_ex5 = [0.3010 0.7450 0.9330];
blueish_ex5 = [193 245 237]/255;
colors_p_ex5 = [linspace(blue_ex5(1),blueish_ex5(1),length_ex5)', linspace(blue_ex5(2),blueish_ex5(2),length_ex5)',linspace(blue_ex5(3),blueish_ex5(3),length_ex5)'];


b_aux_ex5 = b_ex5;
for i = 1:length_ex5
    b_aux_ex5(1:2) = b_ex5(1:2) + 10*i;
    color_ex5 = colors_p_ex5(i, :);
    plotShape(eig_vectors_ex5, b_aux_ex5, mean_shapes_ex5, color_ex5);
    hold on
end
title('Shape generation - first 2 eigenvectors with varying coefficients')
%legend('Constructed shape', 'Mean shape', Location='southeast');




%% Experiment 4 - Choosing a mode and varying it with a noise of +- lambda.

% Random numbers in the interval [0, 4]
b_ex5  = rand(nPoints_ex5  * nDimensions_ex5 , 1);
b_ex5(1) = 5;
b_ex5(2) = 100;
b_std = std(b_ex5);


figure;

for i = 1:10
    noise = rand(nPoints_ex5  * nDimensions_ex5 , 1)* (2*b_std) - b_std;
    b_aux = b_ex5 + noise;
    plotShape(eig_vectors_ex5, b_aux, mean_shapes_ex5, 'blue');
    hold on;
    
end
title('Chosen mode + noise variation of coefficients');
legend('Constructed shape', 'Mean shape', Location='southeast');



%% c) 

nEigenvectors = 13;
stddeviations = 10;
b_ex5 = randn(1, nEigenvectors) .* stddeviations;
b_ex5 = b_ex5';
[mean_shapes_ex5, eig_vectors_ex5, eig_values_ex5] = our_pca(reshaped_data_ex5, nEigenvectors);

disp(size(eig_vectors_ex5));
% figure;
% for i = 1:nShapes_ex5
%    plot(aligned(:, 1, i), aligned(:, 2, i), color='#4DBEEE', LineStyle=':');
%    hold on
% end
% %plot(mean_shapes_ex5(1:nPoints_ex5), mean_shapes_ex5((nPoints_ex5+1):(nPoints_ex5*nDimensions_ex5)), color='red');
% title('Training shapes vs. mean shape');



% Displaying the explained variance covered by the sorted eigenvectors
disp('Explained variance covered by eigenvectors:')
var_exp_ex5  = sort(abs(eig_values_ex5), 'descend') / sum(eig_values_ex5);
for i = 1:nPoints_ex5*nDimensions_ex5
   fprintf('%d: %.2f (%.2f %%)\n', i, var_exp_ex5(i), 100*sum(var_exp_ex5(1:i)));
end 


% Model captures 100% b(1:13)
% b_ex5  = zeros(nPoints_ex5  * nDimensions_ex5 , 1);
b_aux = b_ex5;
b_aux(14:end) = 0;
figure;
plotShape(eig_vectors_ex5, b_aux, mean_shapes_ex5, 'blue');
title('Shape generation - first 13 eigenvectors (100%)');
legend('Constructed shape', 'Mean shape', Location='southeast');


% Model captures 95% b(1:4) - 94.04%
% b_ex5  = zeros(nPoints_ex5  * nDimensions_ex5 , 1);
b_aux = b_ex5;
b_aux(5:end) = 0;
figure;
plotShape(eig_vectors_ex5, b_aux, mean_shapes_ex5, 'blue');
title('Shape generation - first 4 eigenvectors (94.04%)');
legend('Constructed shape', 'Mean shape', Location='southeast');

% Model captures 90% b(1:3) - 90.58%
% b_ex5  = zeros(nPoints_ex5  * nDimensions_ex5 , 1);
b_aux = b_ex5;
b_aux(4:end) = 0;
figure;
plotShape(eig_vectors_ex5, b_aux, mean_shapes_ex5, 'blue');
title('Shape generation - first 3 eigenvectors (90.58%)');
legend('Constructed shape', 'Mean shape', Location='southeast');

% Model captures 80% b(1:2) - 79.28%
% b_ex5  = zeros(nPoints_ex5  * nDimensions_ex5 , 1);
b_aux = b_ex5;
b_aux(3:end) = 0;
figure;
plotShape(eig_vectors_ex5, b_aux, mean_shapes_ex5, 'blue');
title('Shape generation - first 2 eigenvectors (79.28%)');
legend('Constructed shape', 'Mean shape', Location='southeast');














function [shape] = generateShape(eig_vectors, b, mean_shapes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OUTPUT
% shape: result of formula: E*b + m_mean


%%% INPUT
% eig_vectors: the set of eigenvectores for the set of shapes
% b: array used to choose the eigenvectores of interest
% mean_shape: the mean of all shapes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------------------------------------------
%Aim: takes a parameter vector b as input and computes new shapes, where
%the length of b indicates the number of Eigenvectors to be considered for
%shape generation
%--------------------------------------------------------------------------


% Constructing the shape, based on the parameter vector b
shape = eig_vectors * b + mean_shapes;

end
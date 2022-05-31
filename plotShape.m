function [] = plotShape(eig_vectors, b, mean_shapes, col)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OUTPUT


%%% INPUT
% eig_vectors: the set of eigenvectores for the set of shapes
% b: array used to choose the eigenvectores of interest
% mean_shape: the mean of all shapes
% col: color
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------------------------------------------
%Aim: Plots the generated shape and the mean of all shapes.
%--------------------------------------------------------------------------

%figure;

% Creating the shape using the given formula
shape = generateShape(eig_vectors, b, mean_shapes);

size_shape = size(shape);


% Taking the x and y dimensions from the shape (x - first half of the array
% and y - second half)
x_shape = shape(1:size_shape(1)/2);
y_shape = shape((size_shape(1)/2+1):size_shape(1));

% Plotting the constructed shape in blue
plot(x_shape, y_shape, color=col);
hold on

% Decomposing into x and y
x_mean_shape = mean_shapes(1:size_shape(1)/2);
y_mean_shape = mean_shapes((size_shape(1)/2+1):size_shape(1));

% Plotting the mean of the shapes on red
plot(x_mean_shape, y_mean_shape, color='red');

end


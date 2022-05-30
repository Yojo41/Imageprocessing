function plot_ourCov(A);

%creating the figure
figure
plot(A(:,1), A(:,2),'b.')
hold on

%finding the mean of the matrix 
mju = mean(A);

% covariance matrix computed from our function
covA = ourCov(A);

% plotting the mean values
plot(mju(1), mju(2), 'ro');
axis equal, grid on;

alpha = 0.1;
%creating the error ellipse for differend standard deviations
h = error_ellipse(covA, mju, 'conf', 0.683);
set(h, 'Color', 'r', 'LineStyle', '--');
h = error_ellipse(covA, mju, 'conf', 0.954);
set(h, 'Color', 'g', 'LineStyle', '--');
h = error_ellipse(covA, mju, 'conf', 0.997);
set(h, 'Color', 'r', 'LineStyle', '--');

legend('Data','MeanValue','StandardDeviation')
end
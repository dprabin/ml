function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% n = number of features + 1
% m = number of training examples

% theta =[1,2]; % get theta as a matrix (n x 1)
% X = [1 2;1 3;1 4;1 5]; % get input independent variable as matrix (m x n)
% y = [4;5;6;7]; % get output dependent variable as matrix (m x 1)

% 23 finding h functions for each training examples (x(i))
predictions = X * theta;
%size(X), size(theta), size(predictions-y)
sqrErrors = (predictions - y) .^ 2;
J = 1/(2*m) * sum(sqrErrors);

% =========================================================================

end

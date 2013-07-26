function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% theta =[1,2]; % get theta as a matrix (2 x 1)
% X = [1 2;1 3;1 4;1 5]; % get input independent variable as matrix (m x 2)
% y = [4;5;6;7]; % get output dependent variable as matrix (m x 1)

% 20 finding h functions for each training examples (x(i))
predictions = X * theta;
sqrErrors = (predictions - y) .^ 2;
J = 1/(2*m) * sum(sqrErrors);

% =========================================================================

end

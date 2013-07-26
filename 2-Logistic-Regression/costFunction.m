function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
[m,n] = size(X); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%added by Prabin
alpha=1;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%26 hx = 1/(1+e(- theta' X)) %sigmoid function of (theta' * X)

    % calculating cost function
    hx = sigmoid(X * theta);
    err = hx - y;
    %lc=cost that is to be summed up in logistic regression e.g. sum(lc)
    %when y=0, first term zeros out and when y=1, second term zeros out
    lc = -y .* log(hx) - (1-y) .* log(1-hx);
    J = sum(lc) ./ m;

    %grad formula is
    %grad= 1/m * sum((hx-y) * xj)
    %here hx is the sigmoid function hx=sigmoid(X*theta)
    %with this method calculating der and others not necessary
    theta = alpha / m * X' * (hx - y);%first vectorized method
    %theta = alpha / m * ((hx -y)' * X)';%another vectorized method



    grad=theta;

% =============================================================

end

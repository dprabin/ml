function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
[m,n] = size(X); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%18               derivatives of the cost w.r.t. each parameter in theta

    % calculating cost function
    hx = sigmoid(X * theta);
    err = hx - y;
    %lc=cost that is to be summed up in logistic regression e.g. sum(lc)
    %when y=0, first term zeros out and when y=1, second term zeros out
    lc = -y .* log(hx) - (1-y) .* log(1-hx);
    tsq=sum(theta(2:n) .^ 2 );%because we don't add theta0 i.e. theta(1)
    J = 1 / m * sum(lc) + lambda/(2*m) * tsq;
    
    grad = 1/ m * X' * (hx - y);%vectorized method
    %here new lambda/m .* grad(2:n) can't be used;
    %old value of theta must be used
    grad(2:n) = grad(2:n) + lambda/m .* theta(2:n);
    %since theta(J)=1/m*sum(hx-y)*X(j)+lambda/m*theta(j) for j=1:n and +0 for j=0

% =============================================================

end

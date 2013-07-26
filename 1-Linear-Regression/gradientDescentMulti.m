function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % m = number of training examples
    % n = number of features + 1
    % X = input variables; (m x n) matrix
    % y = output variables; (m x 1) matrix
    % alpha = learning rate; a number 
    % theta = coefficients; (n x 1) matrix
    % 26 J = cost function; a number

    n=size(X,2);
    hx = X * theta;
    err = hx - y;
    %err = X * theta - y;
    der=zeros(m,n);
    temp=zeros(n,1);

    for i=1:n
        der(:,i) =  err .* X(:,i);%look at orientation of vectors
    end %inner for
    temp=theta'-alpha*sum(der)/m;
    theta=temp';

    %theta in one go;
    %theta = theta - alpha / m * ((X*theta - y)' * X)';
    %theta = theta - alpha ./ m .* (X' * (X*theta - y));
    %both are similar
    %first one: transpose with less dimensions 2 times
    %second one: transpose with high dimension 1 time

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end %outer for

end

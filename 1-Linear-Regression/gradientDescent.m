function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%prabin
n = size(X,2);
%prabin

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % m = number of training examples
    % n = number of features + 1
    % X = input variables; (m x n) matrix
    % y = output variables; (m x 1) matrix
    % alpha = learning rate; a number 
    % theta = coefficients; (n x 1) matrix
    % 30 J = cost function; a number

    hx = X * theta;
    err = hx - y;
    der=zeros(m,n);
    temp=zeros(n,1);
    der1 = err .* X(:,1);
    der2 = err .* X(:,2);
    %for i=1:n
    %    der(:,i) =  err .* X(:,i);
    %    temp(1,i)=theta(i,1) -alpha* sum(der(:,i))/m;
    %end
    temp(1,1) = theta(1) - alpha * sum(der1) / m;
    temp(2,1) = theta(2) - alpha * sum(der2) / m;
    theta=temp;

    % calculate theta in one go
    %((X*theta-y)' * X)' with no sum
    %X=(97x2);theta=(2x1);X*theta=(97*1);X*theta-y=(97x1)
    %err'=(X*theta-y)'=(1x97);err'*X=(1*2);(err'*X)'=(2*1)=theta
    %theta = theta - alpha / m * ((X*theta - y)' * X)';


    % in two steps
    %temp(1)=theta(1) - alpha / m * sum((X * theta - y) .*X(:,1));
    %temp(2)=theta(2) - alpha / m * sum((X * theta - y) .*X(:,2));
    %theta=temp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

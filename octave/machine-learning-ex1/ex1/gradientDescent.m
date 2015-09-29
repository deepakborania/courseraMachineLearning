function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

tic
for iter = 1:num_iters

    %This raises warning. What was I thinking?
    %theta = theta - (alpha*(1/m)*sum((X*theta - y).*(X)))';
    
    %Using loops : 0.215708 seconds
%    temp = zeros(length(theta),1);
%    for i = 1:length(theta)
%      temp(i) = theta(i) - (alpha*(1/m)*sum((X*theta - y).*X(:,i)));
%    end
%    theta = temp;
    
    %Using bsxfun : 0.103053 seconds
    theta = theta - (alpha*(1/m)*sum(bsxfun(@times,(X*theta - y),X)))';

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
toc
end

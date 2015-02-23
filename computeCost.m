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


h = X*theta; %{multiply X and theta, in the proper order that the inner dimensions match}
error = h-y;%{the difference between h and y}
error_sqr = error.^2; %{use what you have learned}
sum_er=sum([error_sqr]);
J = 1/(2*m) * sum_er; % {multiply 1/(2*m) times the sum of the error_sqr vector}

% =========================================================================

end

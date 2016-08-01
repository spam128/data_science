function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X=[ones(size(X)(1),1) X];
Theta1=[ones(size(Theta1')(1),1) Theta1']';
#theta1 25x401, theta2 10x26
a2=sigmoid(X*Theta1');# m x Sl2 = 401 x 25
a3=sigmoid(a2*Theta2')';#m x Sl3 = 401 x 10
[percent,p]=max(a3);
p=p';




% =========================================================================


end

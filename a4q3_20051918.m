function a4q3
% A4Q3: function for CISC371, Fall 2019, Assignment #4, Question #3

% Load the audio file as a long vector; need the frequency value
[svec fval] = audioread('combo2.wav');

% Regularize the signal
lambda = [52]
wvec = totvar(svec, lambda);

% Play the original vector and the regularized vector;
% a volume correction is included in this example
mval = median(abs(svec))/median(abs(wvec));
sound(svec, fval);
pause(3.5);
sound(mval*wvec, fval);

% Save the result
audiowrite('a4q3out.wav', wvec, fval);
end

function wvec = totvar(yvec, lambda_in)
% WVEC=TOTVAR(YVEC,LAMBDA) performs total variation
% regularization on the 1D signal YVEC, using parameter LAMBDA,
% and returns the result as WVEC
% INPUTS:
%        YVEC   - Mx1 signal vector
%        LAMBDA - non-negative scalar, default is 1
% OUTPUT:
%        WVEC   - the regularized signal

% Problem size
n = numel(yvec);

% Ensure that "lambda" is valid
if nargin >= 2 & ~isempty(lambda_in)
    lambda = max(lambda_in, 1e-6);
else
    lambda = 1;
end
% The regularization matrix - must be sparse - performs differentiation
% %
% % STUDENT CODE: REPLACE SPARSE IDENTITY MATRIX WITH A SPARSE BI-DIAGONAL
% %
onevec = -1*ones(n,1);
R = spdiags([ones(n, 1) onevec], [0,1], n-1, n);
Rsquared = R'*R;

% Regularization is the solution of a linear equation
wvec = (speye(n) + lambda*Rsquared)\yvec;
end

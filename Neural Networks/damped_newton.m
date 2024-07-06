function a2q1_20051918
% Code to test CISC371, Fall 2019, Assignment #2, Question #1

% Create the anonymous functions
a2q1funs;

% Use the same start point and backtracking values for each method
w0 = [-1.2 ; 1];
alpha = 0.5;
beta = 0.5;

% Stepsizes for the methods
sfixed = 0.001;
sline = 0.1;

% Minimizer, minimum, and iterations for fixed stepsize
[wmfixed1, fmfixed1, ixfixed1] = steepfixed(f1fun, w0, sfixed);
[wmfixed2, fmfixed2, ixfixed2] = steepfixed(f2fun, w0, sfixed);
[wmfixed3, fmfixed3, ixfixed3] = steepfixed(f3fun, w0, sfixed);

% Minimizer, minimum, and iterations for line search
[wmline1, fmline1, ixline1] = steepline(f1fun, w0, sline, alpha, beta);
[wmline2, fmline2, ixline2] = steepline(f2fun, w0, sline, alpha, beta);
[wmline3, fmline3, ixline3] = steepline(f3fun, w0, sline, alpha, beta);

% Minimizer, minimum, and iterations for Newton with line search
[wmnewton1, fmnewton1, ixnewton1] = newtondamped(f1fun, f1hess, w0, alpha, beta);
[wmnewton2, fmnewton2, ixnewton2] = newtondamped(f2fun, f2hess, w0, alpha, beta);
[wmnewton3, fmnewton3, ixnewton3] = newtondamped(f3fun, f3hess, w0, alpha, beta);

disp('Fixed stepsize, iterations and minimizer:');
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [1 ixfixed1 wmfixed1']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [2 ixfixed2 wmfixed2']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [3 ixfixed3 wmfixed3']));

disp(sprintf(' '));
disp('Line search, iterations and minimizer:');
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [1 ixline1 wmline1']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [2 ixline2 wmline2']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [3 ixline3 wmline3']));

disp(sprintf(' '));
disp('Damped Newton search, iterations and minimizer:');
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [1 ixnewton1 wmnewton1']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [2 ixnewton2 wmnewton2']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [3 ixnewton3 wmnewton3']));

%Plot the functions for my own sanity.
w1 = linspace(-5,5,50);
w2 = linspace(-5,5,50);
[W1,W2] = meshgrid(w1,w2);
f1 = @(wi,wj) (((wi+1.21) - 2*(wj-1)).^4 + 64*(wi+1.21).*(wj-1));
f2 = @(wi,wj) (2*wj.^3 -6*wj.^2 + 3*wi.^2.*wj);
f3 = @(wi,wj) (100*(wj - wi.^2).^2 + (1 - wi).^2);
%Function 1
figure(1)
mesh(W1,W2,f1(W1,W2))
hold on
plot3(wmfixed1(1),wmfixed1(2),fmfixed1,'Marker','.','Color','red')
plot3(wmline1(1),wmline1(2),fmline1,'Marker','.','Color','green')
plot3(wmnewton1(1),wmnewton1(2),fmnewton1,'Marker','.','Color','black')
legend('f1', 'Fixed Minimizer','Backtracking Minimizer','Newtons Minimizer')
hold off


%Funtion 2
figure(2)
mesh(W1,W2,f2(W1,W2))
hold on
plot3(wmfixed2(1),wmfixed2(2),fmfixed2,'Marker','.','Color','red')
plot3(wmline2(1),wmline2(2),fmline2,'Marker','.','Color','green')
plot3(wmnewton2(1),wmnewton2(2),fmnewton2,'Marker','.','Color','black')
legend('f2', 'Fixed Minimizer','Backtracking Minimizer','Newtons Minimizer')
hold off

%Function 3 
figure(3)
mesh(W1,W2,f3(W1,W2))
hold on
plot3(wmfixed3(1),wmfixed3(2),fmfixed3,'Marker','.','Color','red')
plot3(wmline3(1),wmline3(2),fmline3,'Marker','.','Color','green')
plot3(wmnewton3(1),wmnewton3(2),fmnewton3,'Marker','.','Color','black')
legend('f3', 'Fixed Minimizer','Backtracking Minimizer','Newtons Minimizer')
hold off
end

function [wmin,fmin,ix]=steepfixed(objgradf,w0,s,imax_in,eps_in)
% [WMIN,FMIN,IX]=STEEPFIXED(OBJGRADF,W0,S,IMAX,F)
% estimates the minimum of function and gradient OBJGRADF, beginning
% at point W0 and using constant stepsize S. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         W0       - initial estimate of W
%         S        - stepsize, positive scalar value
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         TMIN     - minimizer, scalar or vector argument for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - Iterations performed

% Set convergence criteria to those supplied, if available
if nargin >= 4 & ~isempty(imax_in)
  imax = imax_in;
else
  imax = 50000;
end

if nargin >= 5 & ~isempty(eps_in)
  epsilon = eps_in;
else
  epsilon = 1e-6;
end

% Initialize: search vector, objective, gradient
wmin = w0;
[fmin gval] = objgradf(wmin);
ix = 0;

while (norm(gval)>epsilon & ix<imax)
    wmin = wmin - s*gval';
    [fmin gval] = objgradf(wmin);
    ix = ix + 1;
end
end

function [wmin,fmin,ix]=steepline(objgradf,w0,s0,alpha,beta,imax_in,eps_in)
% [WMIN,FMIN]=STEEPLINE(OBJGRADF,W0,S,ALPHA,BETA,IMAX,EPS)
% estimates the minimum of function and gradient OBJGRADF, beginning
% at point W0 and using constant stepsize S. Backtracking is
% controlled by comparison ALPHA and reduction ratio BETA. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         W0       - initial estimate of W
%         S        - stepsize, positive scalar value
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         TMIN     - minimizer, scalar or vector argument for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - Iterations performed

% Set convergence criteria to those supplied, if available
if nargin >= 6 & ~isempty(imax_in)
  imax = imax_in;
else
  imax = 50000;
end

if nargin >= 7 & ~isempty(eps_in)
  epsilon = eps_in;
else
  epsilon = 1e-6;
end

% Limit ALPHA and BETA to the interval (0,1)
alpha = max(1e-6, min(1-(1e-6), alpha));
beta  = max(1e-6, min(1-(1e-6), beta));

% Initialize: objective, gradient, unit search vector
wmin = w0;
[fmin gofm] = objgradf(wmin);
dvec = -gofm'/norm(gofm);
ix = 0;

while (norm(gofm)>epsilon & ix<imax)
    % Start with given stepsize and back off exponentially
    s = s0;
    jx = 0;
    [fest, ~] = objgradf(wmin + s*dvec);
    %Implement Armijo's Condition for backtracking.
    while (fest > fmin + alpha*s*norm(gofm).^2*dvec)
        %Update the step size.
        s = s*beta;
        [fest, ~] = objgradf(wmin+s*dvec);
    end
    wmin = wmin + s*dvec;
    [fmin gofm] = objgradf(wmin);
    dvec = -gofm'/norm(gofm);
    ix = ix + 1;
end
end

function [wmin,fmin,ix]=newtondamped(objgradf,hessf,w0,alpha,beta,imax_in,eps_in)
% [WMIN,FMIN,IX]=NEWTONDAMPED(OBJGRADF,HESSF,W0,ALPHA,BETA,IMAX,EPS)
% estimates the minimum of function and gradient OBJGRADF, and Hessian
% matrix HESSF, using a damped Newton's method. It begins at point W0,
% estimates the stepsize by backtracking with ALPHA and BETA subject to
% a modified Armijo condition, and took iterations IX.
%
% Optional arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX.
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         HESSF    - Hessian function  of scalar or vector argument T
%         W0       - initial estimate of W
%         ALPHA    - scalar comparison tolerance, 0<alpha<1
%         BETA     - scalar exponential back-off argument, 0<beta<1
%         IMAX     - optional, limit on iterations; default is 200
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         WMIN  - minimizer, scalar or vector argument for OBJF
%         FMIN  - scalar value of OBJF at, or near, TMIN
%         IX    - number of iterations performed


% Set convergence criteria to those supplied, if available
if nargin >= 6 & ~isempty(imax_in)
  imax = imax_in;
else
  imax = 200;
end

if nargin >= 7 & ~isempty(eps_in)
  epsilon = eps_in;
else
  epsilon = 1e-6;
end

% Initialize: search vector, objective, gradient, Hessian, descent vector
wmin = w0;
[fmin gofm] = objgradf(wmin);
hmat = hessf(wmin);
dvec = hmat\(-gofm');
ix = 0;

while (norm(gofm)>epsilon & ix<imax)
    % Start with unit stepsize and back off exponentially
    s = 1;
    [fest, ~] = objgradf(wmin + s*dvec);
    %Backtracking to estimate an optimal "s" with Armijo's condition.
    while (fest > fmin + alpha*s*norm(gofm).^2*dvec)
        s = s*beta;
        [fest, ~] = objgradf(wmin + s*dvec);
    end
    % Step in the descent direction and re-compute the Netwon variables
    wmin = wmin + s*dvec;
    [fmin gofm] = objgradf(wmin);
    hmat = hessf(wmin);
    dvec = hmat\(-gofm');
    ix = ix + 1;
end
end

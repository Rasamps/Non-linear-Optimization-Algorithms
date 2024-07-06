function a1q1
% CISC371, Fall 2019, Assignment #1, Question #1: scalar optimization

% Anonymous functions for objective functions, gradients, and second
% derivatives
f1 =@(t) exp(3*t)+5*exp(-2*t);
g1 =@(t) 3*exp(3*t)-10*exp(-2*t);
h1 =@(t) 9*exp(3*t)+20*exp(-2*t);
f2 =@(t) log(t).^2 - 2 + log(10-t).^2 - t.^0.2;
g2 =@(t) (2*log(t)/t) - ((2*log(10-t))/(10-t)) - 0.2*(t^-0.8);
h2 =@(t) (4*t^2 - 2*t^2*log(t) - 2*t^2*log(10-t) - 40*t + 40*t*log(t) + 0.16*t^2.2 - 3.2*t^1.2 + 16*t^0.2-200*log(t) + 200)/(t^2*(10-t)^2);
f3 =@(t) -3.*t + sin(0.75.*t).*exp(-2.*t);
g3 =@(t) -3 + 0.75*exp(-2.*t)*cos(0.75*t)-2.*exp(-2.*t)*sin(0.75.*t);
h3 =@(t) 3.4375*exp(-2*t)*sin(0.75.*t) - 3*exp(-2.*t)*cos(0.75*t);

% Unify the above functions for standard calls to optimization code
fg1  =@(t) deal((f1(t)), (g1(t)));
fgh1 =@(t) deal((f1(t)), (g1(t)), (h1(t)));
fg2  =@(t) deal((f2(t)), (g2(t)));
fgh2 =@(t) deal((f2(t)), (g2(t)), (h2(t)));
fg3  =@(t) deal((f3(t)), (g3(t)));
fgh3 =@(t) deal((f3(t)), (g3(t)), (h3(t)));


% Compute the quadratic approximations and backtracking interpolations

%Compute the quadratic approximations first.
[fcoef1,tstat1] = quadapprox(fgh1, 1);
fprintf("The location of the estimated local minimum on f1 using Quadratix Approximation is: %f and the value of the function at this point is: %f.\n",tstat1,f1(tstat1))
[fcoef2,tstat2] = quadapprox(fgh2, 9.9);
fprintf("The location of the estimated local minimum on f2 using Quadratix Approximation is: %f and the value of the function at this point is: %f.\n",tstat2,f2(tstat2))
[fcoef3,tstat3] = quadapprox(fgh3, 0);
fprintf("The location of the estimated local minimum on f3 using Quadratix Approximation is: %f and the value of the function at this point is: %f.\n",tstat3,f3(tstat3))
fprintf("\n")
t1space = linspace(0,1,200);
t2space = linspace(6,9.9,200);
t3space = linspace(-1,2*pi,200);
%Plots the function and the respective estimated local minimizer obtained 
%through Quadratic Approximation.
figure(1)
plot(t1space,f1(t1space))
hold on
plot(tstat1,f1(tstat1),'-o')
hold off
title('f1 and the estimated local minimizer using Quadratic Approximation')
figure(2)
plot(t2space,f2(t2space))
hold on
plot(tstat2,f2(tstat2),'-o')
hold off
title('f2 and the estimated local minimizer using Quadratic Approximation')
figure(3)
plot(t3space,f3(t3space))
hold on
plot(tstat3,f3(tstat3),'-o')
hold off
title('f3 and the estimated local minimizer using Quadratic Approximation')

%Compute the backtracking approximations with one step second
[tmins1,fmins1,ixs1] = steepline(fg1,1,0.1,0.5,0.5,1);
fprintf("Using backtracking line search and taking 1 step ")
fprintf("we get the estimated local minimum on f1 is: %f and the value of the function at this point is: %f.\n",tmins1,fmins1)
[tmins2,fmins2,ixs2] = steepline(fg2,9.9,(9.9-6)/10,0.5,0.5,1);
fprintf("Using backtracking line search and taking 1 step ")
fprintf("we get the estimated local minimum on f2 is: %f and the value of the function at this point is: %f.\n",tmins2,fmins2)
[tmins3,fmins3,ixs3] = steepline(fg3,0,(2*pi)/10,0.5,0.5,1);
fprintf("Using backtracking line search and taking 1 step ")
fprintf("we get the estimated local minimum on f3 is: %f and the value of the function at this point is: %f.\n",tmins3,fmins3)
fprintf("\n")

%Plot the function and its respective local minimizer obtained through
%Backtracking Line Search with one step.
figure(4)
plot(t1space,f1(t1space))
hold on
plot(tmins1,f1(tmins1),'-o')
hold off
title('f1 and the estimated local minimizer using Backtracking Line Search with one step')
figure(5)
plot(t2space,f2(t2space))
hold on
plot(tmins2,f2(tmins2),'-o')
hold off
title('f2 and the estimated local minimizer using Backtracking Line Search with one step')
figure(6)
plot(t3space,f3(t3space))
hold on
plot(tmins3,f3(tmins3),'-o')
hold off
title('f3 and the estimated local minimizer using Backtracking Line Search with one step')

%Compute the backtracking approximations with k steps last.
[tminm1,fminm1,ixm1] = steepline(fg1,1,0.1,0.5,0.5,50000,10^-6);
fprintf("Using backtracking line search and taking k steps, where k is: %f.",ixm1)
fprintf(" we get that the estimated local minimum of f1 is: %f and the value of the function at this point is: %f.\n",tminm1,fminm1)
[tminm2,fminm2,ixm2] = steepline(fg2,9.9,(9.9-6)/10,0.5,0.5,50000,10^-6);
fprintf("Using backtracking line search and taking k steps, where k is: %f.",ixm2)
fprintf(" we get that the estimated local minimum of f2 is: %f and the value of the function at this point is: %f.\n",tminm2,fminm2)
[tminm3,fminm3,ixm3] = steepline(fg3,0,(2*pi)/10,0.5,0.5,50000,10^-6);
fprintf("Using backtracking line search and taking k steps, where k is: %f.",ixm3)
fprintf(" we get that the estimated local minimum of f3 is: %f and the value of the function at this point is: %f.\n",tminm3,fminm3)

%Plot the function with its respective local minimizer obtained through
%Backtracking Line Search with k steps. (Converges by Armijo's condition)
figure(7)
plot(t1space,f1(t1space))
hold on
plot(tminm1,f1(tminm1),'-o')
hold off
title('f1 and the estimated local minimizer using Backtracking Line Search with k steps')
figure(8)
plot(t2space,f2(t2space))
hold on
plot(tminm2,f2(tminm2),'-o')
hold off
title('f2 and the estimated local minimizer using Backtracking Line Search with k steps')
figure(9)
plot(t3space,f3(t3space))
hold on
plot(tminm3,f3(tminm3),'-o')
hold off
title('f3 and the estimated local minimizer using Backtracking Line Search with k steps')

%The following plots pertain to Question 2 of the Assignment.
w1 = linspace(-3,3,200);
w2 = linspace(-3,3,200);
[W1,W2] = meshgrid(w1,w2);

f4 = @(t1,t2) (t1-2*t2).^4 + 64*t1.*t2;
f5 = @(t1,t2) 2*t2.^3 - 6*t2.^2 + 3*(t1.^2).*t2;
f6 = @(t1,t2) 100*(t2-t1.^2).^2 + (1-t1).^2;
%Plot the 3-D mesh plots of the three functions
figure(10)
mesh(W1,W2,f4(W1,W2))
figure(11)
mesh(W1,W2,f5(W1,W2))
figure(12)
mesh(W1,W2,f6(W1,W2))
%Plot the three contour plots of the three functions
figure(13)
contour(W1,W2,f4(W1,W2),'levels',25)
figure(14)
contour(W1,W2,f5(W1,W2),'levels',25)
figure(15)
contour(W1,W2,f6(W1,W2),'levels',25)
end

function [fcoef, tstat] = quadapprox(funfgh, t1)
% [FCOEF,TSTAT]=QUADAPPROX(FUNFGH,T1) finds the polynomial coefficients
% FCOEF of the quadratic approximation of a function at a scalar point
% T1, using the objective value, gradient, and Hessian from unction FUNFGH
% at T1 to complete the approximation. FCOEF is ordered for use in POLYVAL.
% The stationary point of the quadratic is returned as TSTAT.
%
% INPUTS:
%         FUNFGH - handle to 3-output function that computed the
%                  scalar-valued function, gradient, and 2nd derivative
%         T1     - scalar argument
% OUTPUTS:
%         FCOEF  - 1x3 array of polynomial coefficients
%         TSTAT  - stationary point of the approximation
% ALGORITHM:
%     Set up and solve a 3x3 linear equation. If the points are colinear
%     then TSTAT is empty

% Initialize the outputs
fcoef = [];
tstat = [];

% Set up a linear equation with the Vandermonde matrix
%V is the Vandermonde matrix.
V = [t1.^2 t1 1; 2*t1 1 0; 2 0 0];
%This creates our y vector which is the function evaluated at our initial
%point.
[fc1,fc1p,fc1pp] = funfgh(t1);
%An equation of the form x = inv(V)*y
%fcoef is the vector x giving us the coefficients to the approximated
%quadratic.
fcoef = inv(V)*[fc1;fc1p;fc1pp];
%We calculate the location of the estimated minimum using the following eq.
tstat = -fcoef(2)/(2*fcoef(1));
end


function [wmin,fmin,ix]=steepline(objgradf,w0,s0,alpha,beta,imax_in,eps_in)
% [WMIN,FMIN,IX]=STEEPLINE(OBJGRADF,W0,S,ALPHA,BETA,IMAX,EPS)
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
s = s0;
fprev = inf;
[fmin, gofm] = objgradf(wmin);
dvec = -gofm'/norm(gofm);
wmat = wmin;
ix = 0;

%Use a while loop to continuously recompute the necessary variables until
%the norm of the gradient falls below epsilon.
while (norm(gofm)>epsilon & ix<imax)
    s = s0; %Let s be the initial stepsize.
    [ftemp,waste] = objgradf(wmat+s*dvec); %Calculate ftemp as the objective function at (t+sd)
    while ftemp > (fmin + alpha*s*norm(gofm)*dvec) %We use this while loop to update the stepsize by multiplying s with beta.
        s = s*beta;
        [ftemp,waste] = objgradf(wmat+s*dvec); %Each iteration recompute the condition.
    end
    %Re-assign the variables.
    wmat = wmat + s*dvec;
    fprev = fmin;
    [fmin,gofm] = objgradf(wmat);
    dvec = -gofm'/norm(gofm);
    ix = ix + 1;
end
wmin = wmat;

end

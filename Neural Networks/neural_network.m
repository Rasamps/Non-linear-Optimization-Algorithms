function a2q2
%Due to the addition of the optional part of this question. The code will
%take a minute to run and the figures are delayed until this is finished!!!

% TESTEXOR: tests 2d data for the XOR relation

% Load the data
load xexor.txt;
load yexor.txt;
ylabels01 = (yexor + 1)/2;

% Set the size of the ANN: L layers, expanded weight vector
Lnum = 2;
wvecN = (Lnum+1) + Lnum*(size(xexor,1) + 1);

% Set the auxiliary data structures for functions and gradients
global ANNDATA
ANNDATA.lnum = Lnum;
ANNDATA.xmat = [xexor' ones(size(xexor, 2), 1)]';
ANNDATA.yvec = ylabels01(:);
ANNDATA.kval = 1;

% Set the starting point: fixed weight vector
w0 = [ 1 ; 1 ; 0 ; 0 ; 1 ; 0 ; 1 ; 1 ; 1];

% Set the learning rate and related parameters
eta   = 0.001;
imax  = 5000;
gnorm = 1e-4;
%alpha = 0.5;
%beta  = 0.5;

% Original data
%     disp('   ... doing RAW...');
%     % Plot and pause
%     plotclass2d(xexor, yexor);
%     title('Example XOR data');
%     disp('ENTER for next method');
%     pause;

% Builtin neural network
%     disp('   ... doing NET...');
%     net2layer = configure(feedforwardnet(3), xexor, ylabels01);
%     net2layer.trainParam.showWindow = 0;
%     [mlnet, mltrain] = train(net2layer, xexor, ylabels01);
%     mlY = (mlnet(xexor)>0.5)*2 - 1;
%     % Plot and pause
%     plotclass2d(xexor, mlY);
%     title('Builtin NET for XOR data');
%     disp('ENTER for next method');
%     pause;

% Hard-coded linear-sigmoid, 1 hidden layer
    disp('   ... doing ANN response...');
    [wann fann iann,accwann] = steepfixed(@annfun, ...
       w0, eta, imax, gnorm);
    ylabann = annclass(wann)*2 - 1;
    % Plot and pause
    disp(sprintf('ANN (%d), W is', iann));
    disp(wann');
    size(accwann)
    figure(1)
    plotclass2d(xexor, ylabann);
    plotline(wann(4:6),'b');
    plotline(wann(7:9),'b');
    title('Custom ANN for XOR data');
    %Plot the changing weights as vectors in a 3-D plot.
    figure(2)
    plot3(accwann(:,1),accwann(:,2),accwann(:,3),'Color','black')
    hold on
    plot3(accwann(:,4),accwann(:,5),accwann(:,3),'Color','blue')
    plot3(accwann(:,7),accwann(:,8),accwann(:,9),'Color','red')
    legend('The output weights','Weights of 1/2 hidden neuron','Weights of 2/2 hidden neuron')
    title('Optimization of weights with a fixed stepsize');
    hold off
    %The below chunks of code repeat the same work above except with a
    %varying step size for the steepest descent method.
    [wann2 fann2 iann2,accwann2] = steepvary(@annfun, ...
       w0, eta, 0.5, 0.5, imax, gnorm);
    ylabann = annclass(wann)*2 - 1;
    disp(sprintf('Number of steps by steep descent with a varying step size was: %d',iann2));
    disp(wann2');
    size(accwann2)
    %Plot the original XOR data versus the hidden weight vectors obtained
    %with a varying step size.
    figure(3)
    plotclass2d(xexor, ylabann);
    plotline(wann2(4:6),'b');
    plotline(wann2(7:9),'b');
    title('Custom ANN for XOR data with a varying stepsize');
    %Plot the changing weights as vectors of all weights obtained through
    %iteration. Varying step size. Done in a 3-D plot.
    figure(4)
    plot3(accwann2(:,1),accwann2(:,2),accwann2(:,3),'Color','black')
    hold on
    plot3(accwann2(:,4),accwann2(:,5),accwann2(:,3),'Color','blue')
    plot3(accwann2(:,7),accwann2(:,8),accwann2(:,9),'Color','red')
    legend('The output weights','Weights of 1/2 hidden neuron','Weights of 2/2 hidden neuron')
    title('Optimization of weights with a varying stepsize');
    hold off

end

function [fval, gform] = annfun(wvec)
% FUNCTION [FVAL,GFORM]=ANNFUN(WVEC) computes the response of a simple
% neural network that has 1 hidden layer of sigmoids and a linear output
% neuron. WVEC is the initial estimate of the weight vector. FVAL is the
% scalar objective evaluation a GFORM is the gradient 1-form (row vector).
% ANNDATA is a global structure containing data and labels.
%
% INPUTS:
%         WVEC    -  weight vector, output then hidden
% GLOBALS:
%         ANNDATA -  Structure containing
%                    lnum - number of hidden units to compute
%                    xmat - MxN matrix, each row is a data vector
%                    yvec - Mx1 column vector, each label is +1 or -1
% OUTPUTS:
%         FVAL    - 1xM row vector of sigmoid responses

global ANNDATA
% Problem size: original data, intermediate data
[n, m] = size(ANNDATA.xmat);
l = ANNDATA.lnum;

% Separate output weights and hidden weights; latter go into a matrix
wvec2 = wvec(1:(l + 1)); %Output weights?
wvecH = reshape(wvec((l+2):end), n, l); %Hidden weights

% Compute the hidden responses as long row vectors, 1 per hidden neuron;
% then, append 1's to pass the transfer functions to the next layer
phi2mat = (1./(1+exp(-(ANNDATA.xmat'*wvecH))))';    %Sigmoid transfer function
phi2mat(end+1,:) = 1;

% Compute the output transfer function: linear in hidden responses
phi2vec = phi2mat'*wvec2;

% ANN quantization is Heaviside step function of transfer function
q2vec = phi2vec >= 0;

% Residual is difference between label and network output
rvec = ANNDATA.yvec - q2vec;

% Objective is sum of squares of residual errors
fval = 0.5*sum((rvec).^2);

% If required, compute and return the gradient 1-form
if nargout > 1
% Compute the hidden differential responses, with the l+1 row being a zero
% one-form. Use the psi matrix defined in course notes.
[~,ul] = size(phi2mat);
psimat = [phi2mat(1,:).*(ones(1,ul)-phi2mat(1,:)) zeros(1,ul) zeros(1,ul);...
          zeros(1,ul) phi2mat(2,:).*(ones(1,ul)-phi2mat(2,:)) zeros(1,ul);...
          zeros(1,ul) zeros(1,ul) zeros(1,ul);];
%Scale the psi matrix by the output weight vector.
psimat = diag(wvec2'*psimat);
%Using Kron, properly set-up a matrix containing the initial data vectors
%to avoid the use of a for loop.
xhatkron = kron([eye(l);zeros(1,l)], ANNDATA.xmat');

%Compute the Jacobian that's been scaled by the output weight vector.
wJmat = psimat*xhatkron;

% Differentiate the residual error and scale the gradient matrix
grad12mat = -diag(rvec)*[phi2mat' wJmat(1:100,1:3) wJmat(101:200,4:6)];
% Net gradient is the sum of the gradients of each data vector
gform = sum(grad12mat);
end
end

function rvec = annclass(wvec)
% FUNCTION RVEC=ANNCLASS(WVEC) computes the response of a simple neural
% network that has 1 hidden layer of sigmoids and a linear output.
% ANNDATA is a global structure containing data and labels.
%
% INPUTS:
%         WVEC  -  weight vector, output then hidden
% GLOBALS:
%         ANNDATA -  Structure containing
%                    lnum - number of hidden units to compute
%                    xmat - MxN matrix, each row is a data vector
%                    yvec - Mx1 column vector, each label is +1 or -1
% OUTPUT:
%         RVEC  - 1xM vector of thresholded linear responses to data

% Problem size: original data, intermediate data
global ANNDATA
[n, m] = size(ANNDATA.xmat);
l = ANNDATA.lnum;

% Separate output weights and hidden weights; latter go into a matrix
wvec2 = wvec(1:(l + 1));
wvecH = reshape(wvec((l+2):end), n, l);

% Compute the hidden responses as long row vectors, 1 per hidden neuron;
% then, append 1's to pass the transfer functions to the next layer
xfmat = (1./(1+exp(-(ANNDATA.xmat'*wvecH))))';
xfmat(end+1,:) = 1;

% Compute the transfer function: linear in hidden responses
hidxfvec = xfmat'*wvec2;

% ANN response is Heaviside step function of transfer function
rvec = (hidxfvec >= 0);
end

function [wmin,fmin,ix,accWeights]=steepfixed(objgradf,w0,s,imax_in,eps_in)
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
%         accWeights    - All the weights accumulated throughout the
%                         iteration.

% Set convergence criteria to those supplied, if available
if nargin >= 5 & ~isempty(imax_in)
  imax = imax_in;
else
  imax = 50000;
end

if nargin >= 6 & ~isempty(eps_in)
  epsilon = eps_in;
else
  epsilon = 1e-6;
end

% Initialize: search vector, objective, gradient
accWeights = [];
wmin = w0;
[fmin gval] = objgradf(wmin);
ix = 0;

while (norm(gval)>epsilon & ix<imax)
    wmin = wmin - s*gval';
    [fmin gval] = objgradf(wmin);
    %Add the current iterations weights to accWeights
    accWeights = [accWeights; wmin'];
    ix = ix + 1;
end
end

function ph = plotclass2d(dmat, lvec, lw)
% PH=PLOTCLASS(DMAT,LVEC,LW) plots a 2d data set DMAT
% for binary classification using characters in LVEC
%
% INPUTS:
%        DMAT - 2xN data matrix
%        LVEC - Nx1 binary classification
%        LW   - optional scalar, line width for plotting symbols
% OUTPUT:
%        PH   - plot handle for the current figure
% SIDE EFFECTS:
%        Plot into the current window. Low-valued labels are shown
%        as red circles and high-valued labels are shown as
%        blue "+". Optionally, LW is the line width.
%        Axes are slightly adjusted to improve legibility.

% Set the line width
if nargin >= 3 & ~isempty(lw)
  lwid = lw;
else
  lwid = 2;
end

ph = gscatter(dmat(1,:),dmat(2,:),lvec, ...
    'rb', 'o+', [], 'off');
set(ph, 'LineWidth', lwid);
axisadjust(1.1);
end

function ph = plotline(vvec, color, lw)
% PLOTLINE(VVEC,COLOR,LW) plots a separating line
% into an existing figure
% INPUTS:
%        VVEC   - (M+1) augmented weight vector
%        COLOR  - character, color to use in the plot
%        LW   - optional scalar, line width for plotting symbols
% OUTPUT:
%        PH   - plot handle for the current figure
% SIDE EFFECTS:
%        Plot into the current window. 

% Set the line width
if nargin >= 3 & ~isempty(lw)
  lwid = lw;
else
  lwid = 2;
end


% Current axis settings
axin = axis();

% Four corners of the current axis
ll = [axin(1) ; axin(3)];
lr = [axin(2) ; axin(3)];
ul = [axin(1) ; axin(4)];
ur = [axin(2) ; axin(4)];

% Normal vector, direction vector, hyperplane scalar
nvec = vvec(1:2);
dvec = [-vvec(2) ; vvec(1)];
dvec = dvec/norm(dvec);
bval = vvec(3);

% A point on the hyperplane
pvec = -bval*nvec/dot(nvec,nvec);

% Projections of the axis corners on the separating line
clist = dvec'*([ll lr ul ur] - pvec);
cmin = min(clist);
cmax = max(clist);

% Start and end are outside the current plot axis, no problem
pmin = pvec +cmin*dvec;
pmax = pvec +cmax*dvec;

% Plot the line and re-set the axis to its original
hold on;
ph = plot([pmin(1) pmax(1)], [pmin(2) pmax(2)], ...
    strcat(color, '-'), 'LineWidth', lwid);
hold off;
end

function axisadjust(axisexpand)
% AXISADJUST(AXISEXPAND) multiplies the current plot
% ranges by AXISEXPAND.  To increase by 5%, use 1.05
%
% INPUTS:
%         AXISEXPAND - positive scalar multiplier
% OUTPUTS:
%         none
% SIDE EFFECTS:
%         Changes the current plot axis

axvec = axis();
axwdth = (axvec(2) - axvec(1))/2;
axhght = (axvec(4) - axvec(3))/2;
axmidw = mean(axvec([1 2]));
axmidh = mean(axvec([3 4]));
axis([axmidw-axisexpand*axwdth , axmidw+axisexpand*axwdth , ...
      axmidh-axisexpand*axhght , axmidh+axisexpand*axhght]);
end

function [wmin,fmin,ix,accWeights]=steepvary(objgradf,w0,s0,alpha,beta,imax_in,eps_in)
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
%         accWeights    - All the weights accumulated throughout the
%                         iteration.

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
%Limit alpha and beta.
alpha = max(1e-6, min(1-(1e-6), alpha));
beta  = max(1e-6, min(1-(1e-6), beta));
% Initialize: search vector, objective, gradient
    %Also initialize a variable which accumulates all weights computed
    %throughout the iteration.
accWeights = [];
wmin = w0;
[fmin gval] = objgradf(wmin);
dvec = gval'/norm(gval);
ix = 0;
%Loop until either limit is satisfied.
while (norm(gval)>epsilon & ix<imax)
    s = s0; 
    [fest, ~] = objgradf(wmin - s*gval');
    %Inner while loop permits backtracking of the stepsize using Armijo's
    %Condition.
    while (fest > fmin - alpha*s*norm(gval).^2*dvec)
        s = s*beta;
        [fest, ~] = objgradf(wmin-s*dvec);
    end
    wmin = wmin - s*dvec;
    [fmin gval] = objgradf(wmin);
    dvec = gval'/norm(gval);
    %Add the current iterations weights to accWeights.
    accWeights = [accWeights; wmin'];
    ix = ix + 1;
end
end
function [N, T] = GGPrnd(alpha, sigma, tau, T)

%GGPrnd samples points of a generalized gamma process.
% [N, T] = GGPrnd(alpha, sigma, tau, T)
%
% Samples the points of the GGP with Lévy measure
%   alpha/Gamma(1-sigma) * w^(-1-sigma) * exp(-tau*w)
%  
% For sigma>=0, it samples points above the threshold T>0 using the adaptive 
% thinning strategy described in Favaro and Teh (2013).
% -------------------------------------------------------------------------
% INPUTS
%   - alpha: positive scalar
%   - sigma: real in (-Inf, 1)
%   - tau: positive scalar
%   - T: truncation threshold; positive scalar
%
% OUTPUTS
%   - N: points of the GGP
%   - T: threshold
% -------------------------------------------------------------------------
% EXAMPLE
% alpha = 100; sigma = 0.5; tau = 1e-4;
% N = GGPrnd(alpha, sigma, tau, T);

% -------------------------------------------------------------------------
% Reference:
% S. Favaro and Y.W. Teh. MCMC for normalized random measure mixture
% models. Statistical Science, vol.28(3), pp.335-359, 2013.

% Copyright (C) Francois Caron, University of Oxford
% caron@stats.ox.ac.uk
% April 2015
%--------------------------------------------------------------------------

% Check the parameters of the GGP
GGPcheckparams(alpha, sigma, tau);

%% Finite activity GGP
if sigma<0 
    rate = exp( log(alpha) - log(-sigma) + sigma*log(tau) ); 
    K = poissrnd(rate);
    N = gamrnd(-sigma, 1/tau, 1, K);
    N = N(N>0);
    T = 0;
    return;
end

%% Infinite activity GGP
if nargin<4 
    % set the threshold automatically so that we sample of the order Njumps jumps
    % Number of jumps of order alpha/sigma/Gamma(1-sigma) * T^{-sigma} for sigma>0
    % and alpha*log(T) for sigma=0
    if sigma>.1
        Njumps = 20000; % Expected number of jumps
        T = exp(1/sigma*(log(alpha) - log(sigma) - gammaln(1-sigma) - log(Njumps)));
    else        
        T = 1e-10;
        if sigma>0
            Njumps = floor(alpha/sigma/gamma(1-sigma)*T^(-sigma));
        else
            Njumps = floor(-alpha*log(T));
        end
    end
else
    if T<=0
        error('Threshold T must be strictly positive');
    end
    if sigma>1e-3
        Njumps = floor(alpha/sigma/gamma(1-sigma)*T^(-sigma));
    else
        Njumps = floor(-alpha*log(T));
    end
    if Njumps >1e7
        warning('Expected number of jumps = %d - press key if you wish to continue', Njumps);
        pause
    end
end    

% Adaptive thinning strategy
N = zeros(1, ceil(Njumps+3*sqrt(Njumps)));
k = 1;
t=T;
count = 0;
while 1
    e = -log(rand); % Sample exponential random variable of unit rate
    if e > W(t, Inf, alpha, sigma, tau)
        N = N(1:k-1);
        return;
    else
        t_new = inv_W(t, e, alpha, sigma, tau);   
    end
   if tau==0 || (log(rand) < ((-1-sigma) * log(t_new/t)))
        % if tau>0, adaptive thinning - otherwise accept always
        N(k) = t_new;
        k = k + 1;
    end
    t = t_new; 
    count = count+1;        
    if count>10^8 % If too many computions, we lower the threshold T and rerun
        warning('T too small - Its value lower at %f', T/10);
        T = T/10;
        N = GGPrnd(alpha, sigma, tau, T);
        return;
    end
end


end

%% ------------------------------------------------------------------------
% SUBFUNCTIONS
% -------------------------------------------------------------------------

function out = W(t, x, alpha, sigma, tau)

if tau>0
    logout = log(alpha) +  log(1-exp(-tau*(x-t))) + (-1-sigma)*log(t) + (-t*tau) - log(tau) - gammaln(1-sigma);
else
    logout = log(alpha) - gammaln(1-sigma) - log(sigma) + log(t^(-sigma) - x^(-sigma));
end
out = exp(logout);


end

function out = inv_W(t, x, alpha, sigma, tau)

if tau>0
    out = t - 1/tau*log(1-gamma(1-sigma)*x*tau/(alpha*t^(-1-sigma)*exp(-t*tau))); 
else
    logout = -1/sigma * log(t^(-sigma) - sigma*gamma(1-sigma)/alpha*x);
    out = exp(logout);
end

end
function [G, w, w_rem, alpha, sigma, tau] = GGPgraphrnd(alpha, sigma, tau, T)

%GGPgraphrnd samples a GGP graph.
% [G, w, w_rem, alpha, sigma, tau] = GGPgraphrnd(alpha, sigma, tau, T)
%
% -------------------------------------------------------------------------
% INPUTS
%   - alpha: positive scalar
%   - sigma: real in (-Inf, 1)
%   - tau: positive scalar
% Optional input:
%   - T: truncation threshold; positive scalar
%
% OUTPUTS
%   - w: sociability parameters of nodes with at least one connection
%   - wrem: sum of the sociability parameters of nodes with no connection
%   - alpha: Parameter alpha of the GGP
%   - sigma: Parameter sigma of the GGP
%   - tau: Parameter tau of the GGP
% -------------------------------------------------------------------------
% EXAMPLE
% alpha = 100; sigma = 0.5; tau = 1e-4;
% G = GGPgraphrnd(alpha, sigma, tau, T);

% Copyright (C) Francois Caron, University of Oxford
% caron@stats.ox.ac.uk
% April 2015
%--------------------------------------------------------------------------


if length(alpha)==2 % sample alpha
    hyper_alpha = alpha;
    alpha = gamrnd(hyper_alpha(1), 1./hyper_alpha(2));
end
if length(sigma)==2 % sample sigma
    hyper_sigma = sigma;
    sigma = 1 - gamrnd(hyper_sigma(1), 1/hyper_sigma(2));
end
if length(tau)==2 % sample tau
    hyper_tau = tau;
    tau = gamrnd(hyper_tau(1), 1./hyper_tau(2));
end
    
% Sample the graph conditional on the weights w
if nargin==3
    w = GGPrnd(alpha, sigma, tau);
elseif nargin==4
    w = GGPrnd(alpha, sigma, tau, T);
else
    error('Too many inputs')
end

% Samples using the conditional Poisson model
cumsum_w = [0, cumsum(w)];
W_star = cumsum_w(end);  % Total mass of the GGP
D_star = poissrnd(W_star^2); % Total number of directed edges

temp = W_star * rand(D_star, 2);
[~, bin] = histc(temp, cumsum_w);
[ind, ~, ib]  = unique(bin(:));
indlog = false(size(w));
indlog(ind) = true;
w_rem = sum(w(~indlog));    
w = w(ind); 
ib = reshape(ib, size(bin));
G = sparse(ib(:, 1), ib(:, 2), ones(size(ib, 1), 1), length(ind), length(ind));
G = logical(G + G');
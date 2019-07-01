function [G, w1, w1_rem, w2, w2_rem, alpha1, sigma1, tau1, alpha2, sigma2, tau2]...
    = GGPbipgraphrnd(alpha1, sigma1, tau1, alpha2, sigma2, tau2, T)

%GGPbipgraphrnd samples a GGP bipartite graph.
% [G, w1, w1_rem, w2, w2_rem, alpha1, sigma1, tau1, alpha2, sigma2, tau2]...
%    = GGPbipgraphrnd(alpha1, sigma1, tau1, alpha2, sigma2, tau2, varargin)
%
% -------------------------------------------------------------------------
% INPUTS
%   - alpha1: positive scalar
%   - sigma1: real in (-Inf, 1)
%   - tau1: positive scalar
%   - alpha2: positive scalar
%   - sigma2: real in (-Inf, 1)
%   - tau2: positive scalar
% Optional input:
%   - T: truncation threshold; positive scalar
%
% OUTPUTS
%   - w1: sociability parameters of nodes of first type with at least one connection
%   - w1_rem: sum of the sociability parameters of nodes  of first type with no connection
%   - w2: sociability parameters of nodes of second type with at least one connection
%   - w2_rem: sum of the sociability parameters of nodes  of second type with no connection
%   - alpha1: Parameter alpha of the GGP
%   - sigma1: Parameter sigma of the GGP
%   - tau1: Parameter tau of the GGP
%   - alpha2: Parameter alpha of the GGP
%   - sigma2: Parameter sigma of the GGP
%   - tau2: Parameter tau of the GGP
% -------------------------------------------------------------------------
% EXAMPLE
% alpha = 100; sigma = 0.5; tau = 1e-4;
% G = GGPgraphrnd(alpha, sigma, tau, T);

% Copyright (C) Francois Caron, University of Oxford
% caron@stats.ox.ac.uk
% April 2015
%--------------------------------------------------------------------------


if length(alpha1)==2 % sample alpha1
    hyper_alpha = alpha1;
    alpha1 = gamrnd(hyper_alpha(1), 1./hyper_alpha(2));
end
if length(sigma1)==2 % sample sigma1
    hyper_sigma = sigma1;
    sigma1 = 1 - gamrnd(hyper_sigma(1), 1/hyper_sigma(2));
end
if length(tau1)==2 % sample tau1
    hyper_tau = tau1;
    tau1 = gamrnd(hyper_tau(1), 1./hyper_tau(2));
end

if length(alpha2)==2 % sample alpha2
    hyper_alpha = alpha2;
    alpha2 = gamrnd(hyper_alpha(1), 1./hyper_alpha(2));
end
if length(sigma2)==2 % sample sigma2
    hyper_sigma = sigma2;
    sigma2 = 1 - gamrnd(hyper_sigma(1), 1/hyper_sigma(2));
end
if length(tau2)==2 % sample tau2
    hyper_tau = tau2;
    tau2 = gamrnd(hyper_tau(1), 1./hyper_tau(2));
end

    
% Sample the graph conditional on the weights w1 and w2
if nargin==6
    w1 = GGPrnd(alpha1, sigma1, tau1);
    w2 = GGPrnd(alpha2, sigma2, tau2);
elseif nargin==7
    w1 = GGPrnd(alpha1, sigma1, tau1, T);
    w2 = GGPrnd(alpha2, sigma2, tau2, T);
else
    error('Too many inputs')
end

% Samples using the conditional Poisson model
cumsum_w1 = [0, cumsum(w1)];
cumsum_w2 = [0, cumsum(w2)];
W1_star = cumsum_w1(end);  % Total mass of the GGP
W2_star = cumsum_w2(end);  % Total mass of the GGP
D_star = poissrnd(W1_star*W2_star); % Total number of directed edges

temp1 = W1_star * rand(D_star, 1);
temp2 = W2_star * rand(D_star, 1);
[~, bin1] = histc(temp1, cumsum_w1);
[~, bin2] = histc(temp2, cumsum_w2);
G = sparse(bin1, bin2, ones(length(bin1), 1), length(w1), length(w2));
G = logical(G);
deg2 = sum(G, 1);
ind2 = deg2>0;
G = G(:, ind2);
deg1 = sum(G, 2);
ind1 = deg1>0;
G = G(ind1, :);

w1_rem = sum(w1(~ind1));    
w1 = w1(ind1); 
w2_rem = sum(w2(~ind2));    
w2 = w2(ind2);
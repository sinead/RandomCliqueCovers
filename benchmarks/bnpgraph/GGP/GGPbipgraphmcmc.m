function [samples, stats] = GGPbipgraphmcmc(G, modelparam, mcmcparam, verbose)

%GGPbipgraphmcmc runs a MCMC sampler for the GGP bipartite graph model
% [samples, stats] = GGPbipgraphmcmc(G, modelparam, mcmcparam, verbose)
%
% -------------------------------------------------------------------------
% INPUTS
%   - G: sparse logical adjacency matrix  
%   - modelparam: structure of model parameters with the following fields:
%            -  alpha: cell of length 2, corresponding to each type of node.
%               If alpha{i} is scalar, the value of alpha{i}. If vector of length
%               2, parameters of the gamma prior over alpha{i}
%            -  sigma: cell of length 2, corresponding to each type of node.
%               if sigma{i} is scalar, the value of sigma. If vector of length
%               2, parameters of the gamma prior over (1-sigma{i})
%            -  tau: cell of length 2, corresponding to each type of node.
%               if tau{i} scalar, the value of tau. If vector of length
%               2, parameters of the gamma prior over tau
%   - mcmcparam: structure of mcmc parameters with the following fields:
%           - niter: number of MCMC iterations
%           - nburn: number of burn-in iterations
%           - thin: thinning of the MCMC output
%           - hyper.MH_nb: number of MH iterations
%           - hyper.rw_std: standard deviation of the random walk
%           - store_w: logical. If true, returns MCMC draws of w1 and w2
%   - typegraph: type of graph ('undirected' or 'simple')
%   - verbose: logical. If true (default), print information
%
% OUTPUTS
%   - samples: structure with the MCMC samples for the variables
%           - w
%           - w_rem
%           - alpha
%           - logalpha
%           - sigma
%           - tau
%   - stats: structure with summary stats about the MCMC algorithm
%           - rate: acceptance rate of the HMC step at each iteration
%           - rate2: acceptance rate of the MH for the hyperparameters at 
%               each iteration
%
% See also graphmcmc, graphmodel
% -------------------------------------------------------------------------

% Copyright (C) Francois Caron, University of Oxford
% caron@stats.ox.ac.uk
% April 2015
%--------------------------------------------------------------------------

if nargin<4
    verbose = true;
end

% Check G
if ~issparse(G) || ~islogical(G)
    error('Adjacency matrix G must be a sparse logical matrix');
end

% Get parameters alpha1, sigma1, tau1, alpha2, sigma2, tau2
if length(modelparam.alpha{1})==2
    alpha1 = 100*rand;
    estimate_alpha1 = 1;
else
    alpha1 = modelparam.alpha{1};
    estimate_alpha1 = 0;
end
logalpha1 = log(alpha1);
if length(modelparam.sigma{1})==2   
    sigma1 = .5;%2*rand - 1;
    estimate_sigma1 = 1;
else
    sigma1 = modelparam.sigma{1};
    estimate_sigma1 = 0;
end
if length(modelparam.tau{1})==2  
    tau1 = 10*rand;
    estimate_tau1 = 1;
else
    tau1 = modelparam.tau{1};
    estimate_tau1 = 0;
end

if length(modelparam.alpha{2})==2
    alpha2 = 100*rand;
    estimate_alpha2 = 1;
else
    alpha2 = modelparam.alpha{2};
    estimate_alpha2 = 0;
end
logalpha2 = log(alpha2);
if length(modelparam.sigma{2})==2   
    sigma2 = .5;%2*rand - 1;
    estimate_sigma2 = 1;
else
    sigma2 = modelparam.sigma{2};
    estimate_sigma2 = 0;
end
if length(modelparam.tau{2})==2  
    tau2 = 10*rand;
    estimate_tau2 = 1;
else
    tau2 = modelparam.tau{2};
    estimate_tau2 = 0;
end


G = sparse(G);
[n, K] = size(G);
[ind_w1, ind_w2] = find(G);


w2 = exp(randn(K, 1));%ones(K, 1)



w1 = exp(randn(n,1));%ones(n, 1);
w1_rem = exp(randn);%1;
w1_rep = sparse(ind_w1, ind_w2, w1(ind_w1), n, K)';

% Parameters of the MCMC algorithm
niter = mcmcparam.niter;
nburn = mcmcparam.nburn;
thin = mcmcparam.thin;

% To store MCMC samples
n_samples = (niter-nburn)/thin;
if mcmcparam.store_w
    w2_st = zeros(n_samples, K, 'double');
    w1_st = zeros(n_samples, n, 'double');
else
    w1_st = [];
    w2_st = [];
end
w2_rem_st = zeros(n_samples, 1);
w1_rem_st = zeros(n_samples, 1);
alpha2_st = zeros(n_samples, 1);
tau2_st = zeros(n_samples, 1);
sigma2_st = zeros(n_samples, 1);
alpha1_st = zeros(n_samples, 1);
tau1_st = zeros(n_samples, 1);
sigma1_st = zeros(n_samples, 1);


m_w2 = full(sum(G, 1));
m_w1 = full(sum(G, 2)');
tic
for i=1:niter
    if rem(i, 2000)==0 && verbose
        fprintf('i=%i\n', i);
        fprintf('alpha1=%.2f\n', alpha1);
        fprintf('sigma1=%.3f\n', sigma1);
        fprintf('tau1=%.2f\n', tau1); 
        fprintf('alpha2=%.2f\n', alpha2);
        fprintf('sigma2=%.3f\n', sigma2);
        fprintf('tau2=%.2f\n', tau2); 
    end
    
    % Sample U
    Umod = sample_Ucond(w2, w1, ind_w2, ind_w1); % Umod = U-1 to have a sparse vector
    

    % Sample hyperparameters of w2
    [alpha2, logalpha2, sigma2, tau2] =...
        update_hyper(m_w2, Umod, w1, w1_rem, w1_rep', alpha2, logalpha2, sigma2, tau2,...
        estimate_alpha2, estimate_sigma2, estimate_tau2, ...
        modelparam.alpha{2}, modelparam.sigma{2}, modelparam.tau{2},...
        mcmcparam.hyper.rw_std, mcmcparam.hyper.MH_nb);

    % Sample w2
    [w2, w2_rem, w2_rep] = sample_w(m_w2, Umod, w1, w1_rem, w1_rep', alpha2, sigma2, tau2); 
        
    % Sample hyperparameters of w1
    [alpha1, logalpha1, sigma1, tau1] =...
        update_hyper(m_w1, Umod', w2, w2_rem, w2_rep', alpha1, logalpha1, sigma1, tau1,...
        estimate_alpha1, estimate_sigma1, estimate_tau1, ...
        modelparam.alpha{1}, modelparam.sigma{1}, modelparam.tau{1},...
        mcmcparam.hyper.rw_std, mcmcparam.hyper.MH_nb);

    % Sample w1
    [w1, w1_rem, w1_rep] = sample_w(m_w1, Umod', w2, w2_rem, w2_rep', alpha1, sigma1, tau1);

    % Print some information
    if i==10
         time = toc * niter/10;
         hours = floor(time/3600);
         minutes = (time - hours*3600)/60;
         fprintf('-----------------------------------\n')
         fprintf('Start MCMC for bipartite GGP graphs\n')
         fprintf('Nb of nodes: (%d,%d) - Nb of edges: %d\n',size(G, 1),size(G, 2), full(sum(G(:))));
         fprintf('Number of iterations: %d\n', niter)         
         fprintf('Estimated computation time: %.0f hour(s) %.0f minute(s)\n',hours,minutes);
         fprintf('Estimated end of computation: %s \n', datestr(now + time/3600/24));
         fprintf('-----------------------------------\n')
    end
    
    % Store output
    if (i>nburn && rem((i-nburn),thin)==0)
        ind = ((i-nburn)/thin);
        
        if mcmcparam.store_w
            w2_st(ind, :) = w2;
            w1_st(ind, :) = w1;
        end
        w2_rem_st(ind) = w2_rem;
        
        w1_rem_st(ind) = w1_rem;
        alpha2_st(ind) = alpha2;
        tau2_st(ind) = tau2;
        sigma2_st(ind) = sigma2;
        alpha1_st(ind) = alpha1;
        tau1_st(ind) = tau1;
        sigma1_st(ind) = sigma1;
    end
    
end

samples.w1 = w1_st;
samples.w1_rem = w1_rem_st;
samples.w2 = w2_st;
samples.w2_rem = w2_rem_st;
samples.alpha1 = alpha1_st;
samples.logalpha1 = log(alpha1_st);
samples.sigma1 = sigma1_st;
samples.tau1 = tau1_st;
samples.alpha2 = alpha2_st;
samples.logalpha2 = log(alpha2_st);
samples.sigma2 = sigma2_st;
samples.tau2 = tau2_st;

stats=struct();

time = toc;
hours = floor(time/3600);
minutes = (time - hours*3600)/60;
fprintf('-----------------------------------\n')
fprintf('End MCMC for bipartite GGP graphs\n')
fprintf('Computation time: %.0f hour(s) %.0f minute(s)\n',hours,minutes);
fprintf('-----------------------------------\n')
end
  
%% ------------------------------------------------------------------------  
% MAJOR SUBFUNCTIONS  
% -------------------------------------------------------------------------


function Umod = sample_Ucond(w, gamma, ind_w, ind_gamma)

K = length(w);
n = length(gamma);

% Sample U conditional on w and gamma
gamma_w = gamma(ind_gamma).*w(ind_w);
% U = sparse(ind_gamma, ind_w, rexprnd(gamma_w, 1), n, K);
Umod = sparse(ind_gamma, ind_w, rexprnd(gamma_w, 1) - 1, n, K); % Umod = U- 1 to have a sparse matrix
end

function [w, w_rem, w_rep] = sample_w(m_w, Umod, gamma, gamma_rem, gamma_rep, alpha, sigma, tau)

[n, K] = size(Umod);
[ind_gamma, ind_w] = find(Umod);

sum_gamma = sum(gamma);
% gamma_rep = sparse(ind_gamma, ind_w, gamma(ind_gamma), n, K);
gamma_U = gamma_rem + sum_gamma + sum(gamma_rep.* Umod); % sum_i gamma_i * u_ij
w = gamrnd(m_w - sigma, 1./(tau+ gamma_U))';

w_rem = GGPsumrnd(alpha, sigma, tau + sum_gamma + gamma_rem);

w_rep = sparse(ind_gamma, ind_w, w(ind_w), n, K);
end

function [alpha, logalpha, sigma, tau, rate2] =...
        update_hyper(m, Umod, gamma, gamma_rem, gamma_rep, alpha, logalpha, sigma, tau,...
        estimate_alpha, estimate_sigma, estimate_tau, ...
        hyper_alpha, hyper_sigma, hyper_tau, rw_std, MH_nb)
    

K = length(m);    
sum_gamma = sum(gamma) + gamma_rem;
gamma_U = sum_gamma + sum(gamma_rep.* Umod);
for nn=1:MH_nb
    if estimate_sigma
        sigmaprop = 1-exp(log(1-sigma) + rw_std(1)*randn);
    else
        sigmaprop = sigma;
    end
    if estimate_tau
        tauprop = exp(log(tau) + rw_std(2)*randn);
    else
        tauprop = tau;
    end
    if sigmaprop>-1
        if estimate_alpha
            alphaprop = gamrnd(K, 1/(GGPpsi(sum_gamma, 1, sigmaprop, tauprop) ));            
            logalphaprop = log(alphaprop);
        else
            alphaprop = alpha;
            logalphaprop = logalpha;
        end
    else % more stable numerically as alpha can take very large values in that case, we sample alpha2=alpha*tau^sigma
        if estimate_alpha
            alpha2prop = gamrnd(K, 1/( GGPpsi(sum_gamma/tauprop, 1, sigmaprop, 1) ));%exp(log(alpha) + rw_std*randn);
            logalphaprop = log(alpha2prop) - sigmaprop*log(tauprop);                
            alphaprop = exp(logalphaprop);
        else
            alphaprop = alpha;
            logalphaprop = logalpha;
        end
    end
    
    [~, logkappa] = GGPkappa(m, gamma_U, 1, sigma, tau);
    [~, logkappaprop] = GGPkappa(m, gamma_U, 1, sigmaprop, tauprop);
     logaccept = sum(logkappaprop - logkappa);
     
     if estimate_alpha
         logaccept = logaccept ...
             + K * (log(GGPpsi((sum_gamma)/tau, 1, sigma, 1) ) + sigma*log(tau)...
             - log(GGPpsi((sum_gamma)/tauprop, 1, sigmaprop, 1) ) - sigmaprop*log(tauprop) );             
         if hyper_alpha(1)>0
             logaccept = logaccept + hyper_alpha(1)*( logalphaprop - logalpha);
         end
         if hyper_alpha(2)>0
              logaccept = logaccept - hyper_alpha(2) * (alphaprop - alpha);
         end
     else
         logaccept = logaccept ...
             - GGPpsi(sum_gamma, alphaprop, sigmaprop, tauprop) ...
            + GGPpsi(sum_gamma, alpha, sigma, tau);
     end
     if estimate_tau
         logaccept = logaccept ...
             + hyper_tau(1)*( log(tauprop) - log(tau)) - hyper_tau(2) * (tauprop - tau);
     end
     if estimate_sigma
         logaccept = logaccept ...
             + hyper_sigma(1)*( log(1 - sigmaprop) - log(1-sigma)) ...
             - hyper_sigma(2) * (1 - sigmaprop - 1 + sigma);
     end
         
     if isnan(logaccept)
         keyboard
     end

    if log(rand)<logaccept
        alpha = alphaprop;
        logalpha = logalphaprop;
        sigma = sigmaprop;
        tau = tauprop;

    end
end
rate2 = min(1, exp(logaccept));
end  

%% ------------------------------------------------------------------------  
% OTHER SUBFUNCTIONS  
% -------------------------------------------------------------------------

function out = rexprnd(lambda, a)

% Sample from a right truncated distribution of pdf
% lambda * exp(-lambda*x) / (1-exp(-lambda*a)) si x in [0,a], 0 sinon

u = rand(size(lambda));
out = -1./lambda .*log(1 - u.*(1-exp(-lambda*a)));
end
  
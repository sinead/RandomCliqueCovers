% Test function for the Matlab package graphCRM

clear all
close all

fprintf('--- TEST Package graphCRM ---\n\n')

% Add path
addpath('./GGP/')
addpath('./utils/')

% Sample Erdos-Renyi graph
n = 100; p = .05;
obj = graphmodel('ER', n, p);
% G = graphrnd(obj);

% Sample GGP graph
alpha = 20; tau = 1; sigma = .5;
obj2 = graphmodel('GGP', alpha, sigma, tau);
G2 = graphrnd(obj2);
plot_degree(G2, 'or', 2);
% MCMC sampler for GGP graph
hyper_alpha = [0,0]; hyper_tau =  [0,0]; hyper_sigma = [0, 0];
objprior2 =  graphmodel('GGP', hyper_alpha, hyper_sigma, hyper_tau);
niter = 100; nburn = 50; nadapt = niter/4; thin = 2; nchains = 1; verbose = false;
objmcmc2 = graphmcmc(objprior2, niter, nburn, thin, nadapt, nchains);
objmcmc2 = graphmcmcsamples(objmcmc2, G2, verbose);

% Sample GGP bipartite graph
obj3 = graphmodel('GGP', {[20, 1],[20,1]}, sigma, {tau,tau}, 'bipartite');
[G3, w1, w1_rem, w2, w2_rem, alpha1, sigma1, tau1, alpha2, sigma2, tau2] ...
    = graphrnd(obj3);
% MCMC sampler GGP bipartite graph
hyper_alpha = [0,0]; hyper_tau =  {[0,0], [0,0]}; hyper_sigma = [0, 0];
objprior3 =  graphmodel('GGP', hyper_alpha, hyper_sigma, hyper_tau, 'bipartite');
niter = 100; nburn = 50; nadapt = niter/4; thin = 2; nchains = 1; verbose = false;
objmcmc3 = graphmcmc(objprior3, niter, nburn, thin, nadapt, nchains);
objmcmc3 = graphmcmcsamples(objmcmc3, G3, verbose);

fprintf('\n--- TEST COMPLETED ---\n')
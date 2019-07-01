%% General settings
%

clear all

% Add paths
addpath('./GGP/');
addpath('./utils/');

set(0,'DefaultAxesFontSize',14)

% Set the seed
rng(685)

%% Load graph
fileID = fopen(strcat('../../notebooks/enron/enron.tsv'), 'r');
formatSpec =  '%f\t%f';
tblSize = [2 Inf];
adjlist = fscanf(fileID, formatSpec, tblSize)';
n = max(max(adjlist)) + 1;

i = adjlist(:, 1) + 1;
j = adjlist(:, 2) + 1;
v = ones(length(i), 1);
G = sparse(i, j, v, n, n);
G = logical(G + G');

%% Plot adjacency matrix
% figure
%spy(G)
% xlabel('Node id');


%% Posterior inference
%
 
% Parameters of the model
hyper_alpha =[0,0];
hyper_tau = [0,0];
hyper_sigma = [0, 0];
objprior = graphmodel('GGP', hyper_alpha, hyper_sigma, hyper_tau, 'simple');

% Parameters of the MCMC algorithm
niter = 50000;  nburn = 10000; nadapt = niter/4;  thin = 20; nchains = 1;
store_w = false; verbose = false;
    
% Run MCMC
tic
objmcmc = graphmcmc(objprior, niter, nburn, thin, nadapt, nchains, store_w); % Create a MCMC object
objmcmc = graphmcmcsamples(objmcmc, G, verbose); % Run MCMC    
time = toc;


%% Sample last 25 values
[L, L1] = size(objmcmc.samples(1).alpha);
for i = 1:25
    alpha = objmcmc.samples(1).alpha(L + 1 - i);
    sigma = objmcmc.samples(1).sigma(L + 1 - i);
    tau = objmcmc.samples(1).tau(L + 1 - i);
    Gsamp = graphrnd(graphmodel('GGP', alpha, sigma, tau), 1e-6);
    [rid, cid] = find(Gsamp);
    fileID = fopen(strcat('../../notebooks/enron/bnpgraph_runs/enron_', int2str(i), '.tsv'),'w');
    formatSpec = '%d\t%d\n';
    for i = 1:length(rid)
        fprintf(fileID, formatSpec, int32(rid(i)), int32(cid(i)));
    end
    fclose(fileID);
end
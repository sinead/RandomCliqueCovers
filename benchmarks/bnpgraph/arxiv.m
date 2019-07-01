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
fileID = fopen(strcat('../../notebooks/arxiv-qr-gc/arxiv.tsv'), 'r');
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
niter = 40000;  nburn = 0; nadapt = niter/4;  thin = 20; nchains = 1;
store_w = false; verbose = false;
    
% Run MCMC
tic
objmcmc = graphmcmc(objprior, niter, nburn, thin, nadapt, nchains, store_w); % Create a MCMC object
objmcmc = graphmcmcsamples(objmcmc, G, verbose); % Run MCMC    
time = toc;


%% Trace plots and posterior histograms
%

nsamples = length(objmcmc.samples(1).w_rem);
[samples_all, estimates] = graphest(objmcmc, nsamples/2) ; % concatenate chains and returns estimates

col = {'k', 'r', 'b'};
% Trace plots
variables = {'alpha', 'sigma', 'tau', 'w_rem'};
names = {'\alpha', '\sigma', '\tau', 'w_*'};
%for j=1:length(variables)
%    figure
%   for k=1:nchains
%        %plot(thin:thin:niter, objmcmc.samples(k).(variables{j}), 'color', col{k});        
%        %hold on
%    end
%    legend boxoff
%    xlabel('MCMC iterations', 'fontsize', 16);
%    ylabel(names{j}, 'fontsize', 16);
%    box off
%end

%% Obtain mean of parameters
alphastar = mean(objmcmc.samples(1).alpha);
sigmastar = mean(objmcmc.samples(1).sigma);
taustar = mean(objmcmc.samples(1).tau);

%% Sample from GGM model
Gsamp = graphrnd(graphmodel('GGP', alphastar, sigmastar, taustar), 1e-6);
[h2, centerbins, x] = plot_degree(Gsamp); 

%% Write to tsv
%[rid, cid] = find(Gsamp);
%fileID = fopen(strcat('../../notebooks/example/', 'bnpgraph_', graphname, '.tsv'),'w');
%formatSpec = '%d\t%d\n';
%for i = 1:length(rid)
%    fprintf(fileID, formatSpec, int32(rid(i)), int32(cid(i)));
%end
%fclose(fileID);

%% Sample last 25 values
for i = 1:25
    alpha = objmcmc.samples(1).alpha(2001 - i);
    sigma = objmcmc.samples(1).sigma(2001 - i);
    tau = objmcmc.samples(1).tau(2001 - i);
    Gsamp = graphrnd(graphmodel('GGP', alpha, sigma, tau), 1e-6);
    [rid, cid] = find(Gsamp);
    fileID = fopen(strcat('../../notebooks/arxiv-qr-gc/bnpgraph_runs/arxiv_', int2str(i), '.tsv'),'w');
    formatSpec = '%d\t%d\n';
    for i = 1:length(rid)
        fprintf(fileID, formatSpec, int32(rid(i)), int32(cid(i)));
    end
    fclose(fileID);
end
%% BNPgraph package: demo_sparse
% 
% This Matlab script shows empirically the sparsity properties of a range of graph models.
%
% For downloading the package and information on installation, visit the
% <http://www.stats.ox.ac.uk/~caron/code/BNPgraph BNPgraph webpage>.
%
% Reference: F. Caron and E.B. Fox. <http://arxiv.org/abs/1401.1137 Sparse graphs using exchangeable random
% measures.>  arXiv:1401.1137, 2014.
%
% Author: <http://www.stats.ox.ac.uk/~caron/ François Caron>, University of Oxford
%
% Tested on Matlab R2014a.


%% General settings
%

clear all

% Add paths
addpath('./GGP/');
addpath('./utils/');

% Save plots and workspace
saveplots = false;
saveworkspace = false;
if saveplots
    rep = './plots/';
    if ~isdir(rep)
        mkdir(rep);
    end
end

% Set the seed
rng('default')

%% Definition of the graph models
% 

% Model 1: Erdos-Renyi
p = .05;
obj{1} = graphmodel('ER', 100, p);
field{1} = 'n'; % field whose value changes
trial{1} = 2:1:600;
n_samples{1} = 1;
optionrnd{1}={};
% Model 2: Barabasi-Albert
obj{2} = graphmodel('BA', 10);
field{2} = 'n'; % field whose value changes
trial{2} = 2:1:600;
n_samples{2} = 1;
optionrnd{2}={};
% Model 3: Lloyd
sig = 0.1; c = 10; d = 1;
obj{3} = graphmodel('Lloyd', 100, sig, c, d);
field{3} = 'n';
trial{3} = floor(2.^(1:.2:7));
n_samples{3} = 10;
optionrnd{3}={};
% Model 4: GGP (sigma=0)
alpha = 100; tau = 2; sigma = 0;
obj{4} = graphmodel('GGP', alpha, sigma, tau);
field{4} = 'alpha';
trial{4} = 1:.5:130;%5000, tau=5
n_samples{4} = 1;
optionrnd{4}=1e-10;
% Model 5: GGP (sigma=0.5)
alpha = 100; tau = 2; sigma = 0.5;
obj{5} = graphmodel('GGP', alpha, sigma, tau);
field{5} = 'alpha';
trial{5} = 1:.05:40;
n_samples{5} = 1;
optionrnd{5}=1e-5;%-5
% Model 6: GGP (sigma=0.8)
alpha = 100; tau = 2; sigma = 0.8;
obj{6} = graphmodel('GGP', alpha, sigma, tau);
field{6} = 'alpha';
trial{6} = 1:.05:40;
n_samples{6} = 1;
optionrnd{6}=1e-4;%-4

%% Sample graphs of various sizes
%

for k=1:length(obj) % For different models
    fprintf('--- Model %d/%d: %s ---\n', k, length(obj), obj{k}.name)    
    for i=1:length(trial{k}) % Varying number of nodes
        if rem(i, 100)==0
            fprintf('Trial %d/%d \n', i, length(trial{k}));
        end
        obj{k}.param.(field{k}) = trial{k}(i);
        for j=1:n_samples{k} % For different samples
            G = graphrnd(obj{k}, optionrnd{k}); % Sample the graph
            nbnodes{k}(i,j) = size(G, 1);
            nbedges{k}(i,j) = sum(G(:))/2 + trace(G)/2;
            maxdegree{k}(i,j) = max(sum(G));
            degreeone{k}(i,j) = sum(sum(G)==1);
        end
    end
end

%% Some plots
%

% Properties of the plots
plotstyles = {'--k', '--+b', '--xc', 'rs', 'rd', 'ro'};
colorstyle = {'k', 'b', [0,0,.6], [1,0,.5], 'r', [.6,0,0]};
leg = {'ER', 'BA', 'Lloyd', 'GGP (\sigma = 0)', 'GGP (\sigma = 0.5)', 'GGP (\sigma = 0.8)'};
set(0,'DefaultAxesFontSize', 12)
set(0,'DefaultTextFontSize', 16)

% Nb of Edges vs nb of nodes on loglog plot 
pas = .4;
figure('name', 'edgesvsnodesloglog')
for k=1:length(obj)
    h = plot_loglog(nbnodes{k}(:), nbedges{k}(:), plotstyles{k}, pas);
    set(h,'linewidth', 2, 'markersize', 8, 'markerfacecolor', colorstyle{k},'color', colorstyle{k});
    hold on
end
xlabel('Number of nodes', 'fontsize', 16)
ylabel('Number of edges', 'fontsize', 16)
legend(leg,'fontsize', 16, 'location', 'northwest')
xlim([10, 500])
ylim([10, 20000])
legend('boxoff')
if saveplots
    savefigure(gcf, 'edgesvsnodes', rep);
end

%%
%

% Nb of Edges/Nb of nodes squared vs nb of nodes on loglog plot 
pas = 1;
figure('name', 'edgesvsnodesloglog')
for k=1:length(obj)
    ind = nbnodes{k}(:)>0;
    h = plot_loglog(nbnodes{k}(ind), nbedges{k}(ind)./nbnodes{k}(ind).^2, plotstyles{k}, pas);
    set(h,'linewidth', 2, 'markersize', 8, 'markerfacecolor', colorstyle{k},'color', colorstyle{k});
    hold on
end
xlabel('Number of nodes', 'fontsize', 16)
ylabel('Nb of edges / (Nb of nodes)^2', 'fontsize', 16)
legend(leg,'fontsize', 16, 'location', 'southwest')
xlim([3, 700])
legend('boxoff')
if saveplots
    savefigure(gcf, 'edgesvsnodes2', rep);
end

%%
%

% Nb of nodes of degree one versus number of nodes
figure('name', 'degonevsnodes');
for k=1:length(obj)
    h = plot_loglog(nbnodes{k}(:), degreeone{k}(:), plotstyles{k}, pas);
    set(h,'linewidth', 2, 'markersize', 8, 'markerfacecolor', colorstyle{k},'color', colorstyle{k});
    hold on
end
xlabel('Number of nodes', 'fontsize', 16);
ylabel('Number of nodes of degree one', 'fontsize', 16);
legend(leg,'fontsize', 16, 'location', 'northwest');
legend('boxoff');
xlim([10, 500])
ylim([1, 2000])
if saveplots
    savefigure(gcf, 'degreeonevsnodes', rep);
end
% Save workspace
if saveworkspace
    save([rep 'test_stats']);
end
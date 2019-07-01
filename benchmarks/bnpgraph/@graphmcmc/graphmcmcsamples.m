function objmcmc = graphmcmcsamples(objmcmc, G, verbose)           

%GRAPHMCMCSAMPLES runs a MCMC algorithm for posterior inference on graphs
%
% objmcmc = graphmcmcsamples(objmcmc, G)
% -------------------------------------------------------------------------
% INPUTS
%   - objmcmc: an object of the class graphmcmc, containing the graph model
%           specifications and the parameters of the MCMC algorithn
%   - G: sparse binary adjacency matrix
%
% OUTPUT
%   - objmcmc: Updated graphmcmc object with the set of samples
%
% See also GRAPHMCMC, GRAPHMCMC.GRAPHMCMC, GRAPHEST
% -------------------------------------------------------------------------

% Copyright (C) Francois Caron, University of Oxford
% caron@stats.ox.ac.uk
% April 2015
% -------------------------------------------------------------------------

if nargin<3
    verbose = true;
end
 objmodel = objmcmc.prior;   
 switch (objmodel.type)
    case 'GGP'
        switch(objmodel.typegraph)
            case {'undirected', 'simple'}
                for k=1:objmcmc.settings.nchains % Run MCMC algorithms
                    fprintf('-----------------------------------\n')
                    fprintf('           MCMC chain %d/%d        \n', k, objmcmc.settings.nchains);
                    [objmcmc.samples(k), objmcmc.stats(k)] = GGPgraphmcmc(G, objmodel.param, objmcmc.settings, objmodel.typegraph, verbose);                        
                end
            case 'bipartite'
                for k=1:objmcmc.settings.nchains % Run MCMC algorithms
                    fprintf('-----------------------------------\n')
                    fprintf('           MCMC chain %d/%d        \n', k, objmcmc.settings.nchains);
                    [objmcmc.samples(k), objmcmc.stats(k)] = GGPbipgraphmcmc(G, objmodel.param, objmcmc.settings, verbose);                        
                end
            otherwise
                error('Unknown type of graph %s', objmodel.typegraph);
        end
    otherwise
        error('Inference not implemented for graph model of type %s', objmodel.type);
 end           
end
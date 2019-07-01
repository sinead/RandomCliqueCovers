function [samples_all, estimates] = graphest(objmcmc, nburn)            

%GRAPHEST returns median estimates of the graph parameters
% [samples_all, estimates] = GRAPHEST(objmcmc, nburn)
%
% -------------------------------------------------------------------------
% INPUTS
%   - objmcmc: an object of the class graphmcmc
% Optional input
%   - nburn: number of MCMC iterations to remove to get estimates (default:0)
%
% OUTPUTS
%   - samples_all: structure containing the MCMC samples for all variables,
%       concatenated over all MCMC chains
%   - estimates: structure containing the median estimates for the
%       different parameter (one field per parameter)
%
% See also GRAPHMCMC, GRAPHMCMC.GRAPHMCMC, GRAPHMCMCSAMPLES
% -------------------------------------------------------------------------

% Copyright (C) Francois Caron, University of Oxford
% caron@stats.ox.ac.uk
% April 2015
% -------------------------------------------------------------------------

names = fieldnames(objmcmc.samples(1));
if isfield(objmcmc.samples(1), 'alpha') %non bipartite
    nsamples = length(objmcmc.samples(1).alpha);
else % bipartite
    nsamples = length(objmcmc.samples(1).alpha1);
end

if nargin<2
    nburn = 0;
end
nsamples_all = nsamples - nburn;


for i=1:length(names)
    if isempty(objmcmc.samples(1).(names{i}))
        samples_all.(names{i}) = [];
        estimates.(names{i}) = [];
    else
        for k=1:objmcmc.settings.nchains
            samples_all.(names{i})((k-1)*nsamples_all+1:k*nsamples_all, :) = objmcmc.samples(k).(names{i})(nburn+1:nsamples, :);                   
        end
        estimates.(names{i}) = median(samples_all.(names{i}));
    end
end  
end
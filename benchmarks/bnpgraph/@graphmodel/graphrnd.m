
function [G, varargout] = graphrnd(objmodel, varargin)

%GRAPHRND samples a graph from a graph model
% [G, varargout] = GRAPHRND(obj, varargin) 
%
% -------------------------------------------------------------------------
% INPUTS
%   - objmodel: an object of the graphmodel class
%
% Optional inputs:
%   - T: truncation threshold (for a GGP graph model)
% -------------------------------------------------------------------------
% OUTPUTS
%   - G: sparse logical adjacency matrix
% 
% Specific outputs:
% For GGP non bipartite:
%   [G, w, w_rem, alpha, sigma, tau] = GRAPHRND(objmodel, T)
%   - w: sociability parameters of nodes with at least one connection
%   - wrem: sum of the sociability parameters of nodes with no connection
%   - alpha: Parameter alpha of the GGP
%   - sigma: Parameter sigma of the GGP
%   - tau: Parameter tau of the GGP
% For GGP bipartite
%   - w: sociability parameters of nodes with at least one connection
%   - wrem: sum of the sociability parameters of nodes with no connection
%   - alpha: Parameter alpha of the GGP
%   - sigma: Parameter sigma of the GGP
%   - tau: Parameter tau of the GGP  
%
% -------------------------------------------------------------------------
%
% See also GRAPHMODEL, GRAPHMODEL/GRAPHMODEL

% Copyright (C) Francois Caron, University of Oxford
% caron@stats.ox.ac.uk
% April 2015
%--------------------------------------------------------------------------

varargout = [];
switch(objmodel.type)
    case 'ER'
        G = ERgraphrnd(objmodel.param.n, objmodel.param.p);                    
    case 'GGP'
        if nargin>2
            error('Only two possible inputs for objects of type GGP')
        end
        switch(objmodel.typegraph)
            case 'undirected'
                [G, w, w_rem, alpha, sigma, tau] = GGPgraphrnd(objmodel.param.alpha, objmodel.param.sigma, objmodel.param.tau, varargin{:});
                varargout = {w, w_rem, alpha, sigma, tau};
            case 'simple'
                [G, w, w_rem, alpha, sigma, tau] = GGPgraphrnd(objmodel.param.alpha, objmodel.param.sigma, objmodel.param.tau, varargin{:});
                varargout = {w, w_rem, alpha, sigma, tau};
                G = G - diag(diag(G));
            case 'bipartite'
                [G, w1, w1_rem, w2, w2_rem, alpha1, sigma1, tau1, alpha2, sigma2, tau2] ...
                    = GGPbipgraphrnd(objmodel.param.alpha{1}, objmodel.param.sigma{1}, objmodel.param.tau{1}, objmodel.param.alpha{2}, objmodel.param.sigma{2}, objmodel.param.tau{2}, varargin{:});
                varargout = {w1, w1_rem, w2, w2_rem, alpha1, sigma1, tau1, alpha2, sigma2, tau2};
        end                    
    case 'BA'
        G = BAgraphrnd(objmodel.param.n);
    case 'Lloyd'
        G = Lloydgraphrnd(objmodel.param.n, objmodel.param.sig, objmodel.param.c, objmodel.param.d);                 
end     

end        

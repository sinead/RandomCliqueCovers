classdef graphmodel
    % graphmodel class containing the parameters of the graph model
    %
    % PROPERTIES
    %   - <a href="matlab: help graphmcmc/name">name</a>
    %   - <a href="matlab: help graphmcmc/type">type</a>
    %   - <a href="matlab: help graphmcmc/param">param</a>
    %   - <a href="matlab: help graphmcmc/typegraph">typegraph</a>
    %
    % CLASS CONSTRUCTOR
    %   - <a href="matlab: help graphmodel/graphmodel">graphmodel</a>
    %
    % METHODS
    %   - <a href="matlab: help graphrnd/graphmcmcsamples">graphmcmcsamples</a>: runs a MCMC algorithm for posterior inference on graphs
    
    properties
        name; % name of the graph model (string)      
        type; % type of graph (string)
        param; % parameters (structure)   
        typegraph = 'undirected'; % undirected, simple (undirected, no loops, no multiple edges), bipartite
    end
    
    %---------------------------------------------------------------------
    methods
        
        %% Class constructor
        function obj = graphmodel(type, varargin)
            %GRAPHMODEL creates a graph model.
            % obj = GRAPHMODEL(type, varargin)
            %
            % Possible type: 'ER', 'GGP', 'BA', 'Lloyd'
            % -------------------------------------------------------------
            % obj = GRAPHMODEL('ER', n, p) creates an Erdos-Rényi graph
            % model with n nodes and probability of connection p in [0,1]
            % -------------------------------------------------------------
            % obj = GRAPHMODEL('GGP', alpha, sigma, tau, typegraph) creates
            % a GGP graph model where
            %   - optional input typegraph can be undirected (default),
            %   simple or bipartite
            %   - If typegraph is undirected or simple:
            %           - alpha: double or vector of length 2. 
            %           In the first case, alpha is supposed to be fixed. 
            %           Otherwise, it is random and drawn from a gamma 
            %           distribution of parameters alpha(1) and alpha(2) 
            %           - sigma: double or vector of length 2. 
            %           In the first case, sigma is supposed to be fixed. 
            %           Otherwise, it is random and (1-sigma) is drawn from
            %           a gamma distribution of parameters sigma(1) and sigma(2) 
            %           - tau: double or vector of length 2. 
            %           In the first case, tau is supposed to be fixed. 
            %           Otherwise, it is random and drawn from a gamma 
            %           distribution of parameters tau(1) and tau(2) 
            %   - If typegraph is bipartite:
            %        - alpha: cell of length 2.     
            %              - alpha{1} corresponds to the parameter alpha of     
            %                   the GGP associated to the first type of node. 
            %                   It may be a double or a vector of length 2. 
            %                   (See above for details)
            %              - alpha{2} corresponds to the parameter alpha of     
            %                   the GGP associated to the second type of node. 
            %                   It may be a double or a vector of length 2. 
            %                   (See above for details)
            %        - sigma: cell of length 2. Same as above.
            %        - tau: cell of length 2. Same as above.
            % -------------------------------------------------------------
            % obj = GRAPHMODEL('BA', n) creates a Barabasi-Albert graph
            % model with n nodes 
            % -------------------------------------------------------------
            % obj = GRAPHMODEL('Lloyd', n, sig, c, d) creates a Lloyd graph
            % model with parameters n, sig, c and d 
            % -------------------------------------------------------------
            % EXAMPLES
            % n = 1000; p = 0.01;
            % obj = graphmodel('ER', n, p);
            % alpha = 100; sigma = 0.1; tau = 1;
            % obj2 = graphmodel('GGP', alpha, sigma, tau);
            % obj3 = graphmodel('BA', n);
            % hyper_alpha = [100, 1]; sigma = 0.1; tau = 1;
            % obj4 = graphmodel('GGP', hyper_alpha, sigma, tau);
            % alpha = {100,50}; sigma = {-1,.5}; tau = {[1,.1],1};
            % obj5 = graphmodel('GGP', alpha, sigma, tau, 'bipartite');
            
            % Copyright (C) Francois Caron, University of Oxford
            % caron@stats.ox.ac.uk
            % April 2015
            %--------------------------------------------------------------
            
            % Check the inputs are correct
            checkparamsgraph(type, varargin{:});
            
            % Creates a graph model
            obj.type = type;
            switch(type)
                case 'ER'
                    obj.param.n = varargin{1};
                    obj.param.p = varargin{2};
                    obj.name = 'Erdos-Renyi';                    
                case 'GGP'                          
                    obj.name = 'Generalized gamma process';                    
                    obj.param.alpha = varargin{1};
                    obj.param.sigma = varargin{2};
                    obj.param.tau = varargin{3};
                    if nargin==5
                        obj.typegraph = varargin{4};   
                        if strcmp(obj.typegraph, 'bipartite')
                            if isnumeric(varargin{1})
                                obj.param.alpha = cell(1, 2);
                                obj.param.alpha{1} = varargin{1};
                                obj.param.alpha{2} = varargin{1};
                            end
                            if isnumeric(varargin{2})
                                obj.param.sigma = cell(1, 2);
                                obj.param.sigma{1} = varargin{2};
                                obj.param.sigma{2} = varargin{2};
                            end
                            if isnumeric(varargin{3})
                                obj.param.tau = cell(1, 2);
                                obj.param.tau{1} = varargin{3};
                                obj.param.tau{2} = varargin{3};
                            end                            
                        end
                    end
                case 'BA'                    
                    obj.param.n = varargin{1};
                    obj.name = 'Barabasi-Albert';
                case 'Lloyd'
                    obj.name = 'Lloyd';
                    obj.param.n = varargin{1};
                    obj.param.sig = varargin{2};
                    obj.param.c = varargin{3};
                    obj.param.d = varargin{4};
                otherwise
                    error('Unknown type %s', type);                    
            end               
        end

        %% Sample a graph
        [G, varargout] = graphrnd(obj, varargin)
    end    
end




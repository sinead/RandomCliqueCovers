classdef graphmcmc    
    % graphmcmc class of MCMC parameters and samples for inference on graph
    %
    % PROPERTIES
    %   - <a href="matlab: help graphmcmc/prior">prior</a>
    %   - <a href="matlab: help graphmcmc/settings">settings</a>
    %   - <a href="matlab: help graphmcmc/samples">samples</a>
    %   - <a href="matlab: help graphmcmc/stats">stats</a>
    %
    % CLASS CONSTRUCTOR
    %   - <a href="matlab: help graphmcmc/graphmcmc">graphmcmc</a>
    %
    % METHODS
    %   - <a href="matlab: help graphmcmc/graphmcmcsamples">graphmcmcsamples</a>: runs a MCMC algorithm for posterior inference on graphs
    %   - <a href="matlab: help graphmcmc/graphest">graphest</a>: returns median estimates of the graph parameters    

    properties
        % Object of class graphmodel
        % Default: graphmodel('GGP', [0, 0], [0, 0], [0, 0]);
        prior = graphmodel('GGP', [0, 0], [0, 0], [0, 0]);
        
        % Settings of the MCMC algorithm
        % Default: struct('niter', 1000, 'nburn', 500, 'thin', 1, 'nchains', 1, 'store_w', 'true',...
        %             'hyper', struct('rw_std', [.02, .02], 'MH_nb', 2));
        settings = struct('niter', 1000, 'nburn', 500, 'thin', 1, 'nchains', 1, 'store_w', 'true',...
            'hyper', struct('rw_std', [.02, .02], 'MH_nb', 2));
        
        % MCMC Samples
        samples;
        
        % Statistics on the MCMC algorithm
        stats;
    end    
    
    %% METHODS    
    methods
        
        % Class constructor
        function obj = graphmcmc(objmodel, niter, nburn, thin, nadapt, nchains, store_w)
            
        %GRAPHMCMC creates an object of class graphmcmc
        % obj = GRAPHMCMC(objmodel, niter, nburn, thin, nadapt, nchains, store_w) 
        %
        % -------------------------------------------------------------------------
        % INPUTS
        %   - objmodel: an object of class graph model
        % Optional inputs
        %   - niter: number of MCMC iterations (default: 1000)
        %   - nburn: number of burn-in iterations (default:nburn/2)
        %   - thin: thinning of the MCMC output (default: 1)
        %   - nadapt: number of iterations for adaptations (default:nburn/2)
        %   - nchains: number of MCMC chains (default: 1)
        %   - store_w: logical. true if we want to store and return samples for w
        %                   (default: true)
        %
        % OUTPUT
        %   - objmcmc: an object of the graphmcmc class
        %
        % See also GRAPHMCMC, GRAPHMODEL, GRAPHMODEL.graphmodel, GRAPHMCMCSAMPLES, GRAPHEST  
        %
        % -------------------------------------------------------------------------
        % EXAMPLE
        % objmodel = graphmodel('GGP', 100, 0, 1);
        % niter = 10000; nburn = 1000; thin = 10; nadapt = 500; nchains = 3
        % objmcmc = graphmcmc(objmodel, niter, nburn, thin, nadapt, nchains)

        % Copyright (C) Francois Caron, University of Oxford
        % caron@stats.ox.ac.uk
        % April 2015
        % -------------------------------------------------------------------------
            
            
            if ~isa(objmodel, 'graphmodel')
                error('First argument must be a model of class graphmodel');
            end
            obj.prior = objmodel;
            switch(objmodel.type)
                case 'GGP'
                    switch(objmodel.typegraph)
                        case {'undirected','simple'}
                            % Settings of the MCMC algorithm
                            obj.settings.leapfrog = struct('L', 5, 'epsilon', .1, 'nadapt', 250);
%                             obj.settings.hyper = struct('rw_std', [.02, .02], 'MH_nb', 2);
                            obj.settings.latent = struct('MH_nb', 1);
                            % Samples
                            obj.samples = struct('w', [], 'w_rem', [], ...
                                'alpha', [], 'logalpha', [], 'sigma', [], 'tau', []);
                            % stats
                            obj.stats = struct('rate', [], 'rate2', []);
                        case 'bipartite'
                            % Samples 
                            obj.samples = struct('w1', [], 'w1_rem', [], ... 
                                'w2', [], 'w2_rem', [], ... 
                                'alpha1', [], 'logalpha1', [], 'sigma1', [], 'tau1', [],...
                                'alpha2', [], 'logalpha2', [], 'sigma2', [], 'tau2', []);
                            % stats
                            obj.stats = struct();
                        otherwise
                            error('Inference not supported for graph of type %s %s', objmodel.type, objmodel.typegraph);
                    end
                    
                otherwise
                    error('Inference not supported for graph of type %s', objmodel.type);
            end
            
            
            if nargin>1
                obj.settings.niter = niter;
                if nargin>2
                    obj.settings.nburn = nburn;
                    if nargin>3
                        obj.settings.thin = thin;            
                    end                                     
                else
                    obj.settings.nburn = floor(niter/2);
                end
                if nargin>4
                    obj.settings.leapfrog.nadapt = nadapt;
                else
                    obj.settings.leapfrog.nadapt = floor(obj.settings.nburn/2);
                end   
                    
            end
            if nargin>5
                obj.settings.nchains = nchains;
            end
            if nargin>6
                obj.settings.store_w = store_w;
            end
        end
        
        % Runs a MCMC sampler
        objmcmc = graphmcmcsamples(objmcmc, G, varargin)      

        
        % Returns estimates from the MCMC output
        [samples_all, estimates] = graphest(objmcmc, nburn)    
        
    end
end
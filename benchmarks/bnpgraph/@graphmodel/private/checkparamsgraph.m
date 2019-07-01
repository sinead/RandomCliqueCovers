function checkparamsgraph(type, varargin)
    
% Checks the validity of the parameters for the graph model

switch(type)
    case 'ER'
        if length(varargin)~=2
            error('Erdos-Renyi graph must have two arguments n and p');
        end
        
        n = varargin{1};
        p = varargin{2};
        if ~isnumeric(n) || (floor(n)-n)~=0
            error('First parameter n must be an integer');
        end
        if ~isnumeric(p) || p>1 || p<0
            error('Second parameter p must be a real in (0,1)')
        end
        return;        
    case 'GGP'
        if length(varargin)>4
            error('GGP graph must have at most 4 arguments');
        end
        
        names_var = {'alpha', 'sigma', 'tau'};
        
        if length(varargin)==4
            typegraph = varargin{4};
            switch(typegraph)
                case {'undirected', 'simple', 'bipartite'}
                otherwise
                    error('Unknown type of graph %s', typegraph);
            end
        end
        
        if (length(varargin)<4 || strcmp(typegraph, 'simple') || strcmp(typegraph, 'undirected'))
            alpha = varargin{1};
            sigma = varargin{2};
            tau = varargin{3};
            [alpha, sigma, tau] = checkalphasigmatau(alpha, sigma, tau);
            GGPcheckparams(alpha, sigma, tau); 
        else
            for i=1:3
                if iscell(varargin{i})
                    eval([names_var{i} '1=varargin{' num2str(i) '}{1};']); 
                    eval([names_var{i} '2=varargin{' num2str(i) '}{2};']);                    
                elseif isnumeric(varargin{i})
                    eval([names_var{i} '1=varargin{' num2str(i) '};']); 
                    eval([names_var{i} '2=varargin{' num2str(i) '};']);
                else
                    error('Argument must be either a cell or a numeric');
                end
            end
            [alpha1, sigma1, tau1] = checkalphasigmatau(alpha1, sigma1, tau1);
            GGPcheckparams(alpha1, sigma1, tau1); 
            [alpha2, sigma2, tau2] = checkalphasigmatau(alpha2, sigma2, tau2);
            GGPcheckparams(alpha2, sigma2, tau2);
        end        
    case 'BA'
        if length(varargin)~=1
            error('BA graph must have one parameter n');
        end
        n = varargin{1};
        if ~isnumeric(n) || (floor(n)-n)~=0
            error('Parameter n must be an integer');
        end
    case 'Lloyd'
        n = varargin{1};
        sig = varargin{2};
        c = varargin{3};
        d = varargin{4};
        if ~isnumeric(n) || ~isnumeric(sig) || ~isnumeric(c) || ~isnumeric(d)
            error('Parameters must be numeric')
        end
        if n<=0 || sig<=0 || c<=0 || d<=0
            error('Parameters must be strictly positive');
        end
    otherwise
        error('Unknown graph type %s\nValid types are ER, GGP, BA, Lloyd', type);
        
end   
end


function [alpha, sigma, tau] = checkalphasigmatau(alpha, sigma, tau)

if length(alpha)==2
    if alpha(1)<0 || alpha(2)<0
        error('Hyperparameters for alpha must be positive');
    end
    alpha = 1;
end
if length(sigma)==2
    if sigma(1)<0 || sigma(2)<0
        error('Hyperparameters for sigma must be positive');
    end
    sigma = 0;
end
if length(tau)==2
    if tau(1)<0 || tau(2)<0
        error('Hyperparameters for tau must be positive');
    end
    tau = 1;
end
end
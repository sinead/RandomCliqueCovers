import numpy as np
import matplotlib.pyplot as plt
import copy
import pdb
import scipy.stats as ss
from scipy.special import gammaln, gamma
import networkx as nx
import pickle
import optim as opt
from scipy.sparse import csr_matrix

def ord(i, j):
    return (i, j) if i < j else (j, i)

def init_nodetocliques(cliques, num_nodes):
    ans = [set() for _ in range(num_nodes)]
    for i, cl in enumerate(cliques):
        for node in cl:
            ans[node].add(i)
    return ans

def init_linktocliques(cliques, links):
    ans = {l: set() for l in links}
    for i, cl in enumerate(cliques):
        for node1 in cl:
            for node2 in cl:
                if node1 < node2:
                    ans[(node1, node2)].add(i)
    return ans

def links_init(links_):
    links = set()
    for i, j in links_:
        if i != j:
            links.add(ord(i, j))
    links = list(links)
    return links
 
def randchoice(x):
    return list(x)[np.random.choice(len(x))]

def poissonparams(K, alpha, sigma, c):
    # vectorised for speed
    ivec = np.arange(K, dtype=float)
    lpp = np.log(alpha) \
        + gammaln(1.0 + c) \
        - gammaln(c + sigma) \
        + gammaln(ivec + c + sigma) \
        - gammaln(ivec + 1.0 + c)
              
    pp = np.exp(lpp)

    return pp

class NetworkFull(object):
    def __init__(self, links, alpha, sigma, lamb, c):
        self.alpha = alpha
        self.sigma = sigma
        self.lamb = lamb
        self.c = c

        # initialize with as many cliques as nodes
        # assuming nodes are in {0,...,1}
        self.links = links_init(links)
        self.num_links = len(self.links)
        self.num_cliques = len(self.links) # num cliques
        self.num_nodes = max([l[1] for l in self.links]) + 1
        self.cliquetonodes = [set(l) for l in self.links] # Z in sparse mode with set entries   
        self.nodetocliques = init_nodetocliques(self.cliquetonodes, self.num_nodes) # Z.t()
        self.nodecount = np.array([len(x) for x in self.nodetocliques], dtype=int)
        self.linktocliques = init_linktocliques(self.cliquetonodes, self.links) # hashed for quickly query, since there is no obvious ordering

        self.alpha_params = [1.,1.]
        self.sigma_params = [1.,1.]
        self.c_params = [1., 1.]
        self.lambda_params = [1.,1.]

        self.alpha_hist = []
        self.sigma_hist = []
        self.c_hist = []
        self.lambda_hist = []
        self.num_cliques_hist = []
        self.lp_hist = []
    
    def sethypers(self, alpha_params=None, sigma_params = None, c_params = None, lambda_params = None):
        if alpha_params is not None:
            self.alpha_params = alpha_params
        if sigma_params is not None:
            self.sigma_params = sigma_params
        if c_params is not None:
            self.c_params = c_params
        if lambda_params is not None:
            self.lambda_params = lambda_params

    def fit(self, num_iters = 10, num_sm = 10, num_optim = 10, save_every = 100,
            write_every = 1000, optim_momentum=0.3, optim_backtracktol=0.1, optim_stepsize=0.01,
            hyper_start = 0, filename = 'sdcheckpoint.pkl', verbose=True):

        for i in range(num_iters):
            self.splitmerge(N = num_sm)
            self.optim(num_iters = num_optim, stepsize = optim_stepsize, momentum = optim_momentum, backtracktol = optim_backtracktol)
            self.samplelamb()

            if i % save_every == 0:
                self.savestate()
                if verbose:
                    print('iter: {:5d}, K: {:5d}, alpha: {:8.3f}, sigma: {:0.3f}, c: {:8.3f}, lamb: {:5.0f}, lp: {:12.3f}'.
                        format(i, self.num_cliques, self.alpha, self.sigma, self.c, self.lamb, self.loglik()))
                    
            if i % write_every == 0:
                if verbose:
                    print('writing to file...')
                with open(filename, 'wb') as fn:
                    pickle.dump(self, fn)
                if verbose:
                    print('done!')
    
    def savestate(self):
        self.alpha_hist.append(self.alpha)
        self.sigma_hist.append(self.sigma)
        self.c_hist.append(self.c)
        self.lambda_hist.append(self.lamb)
        self.num_cliques_hist.append(self.num_cliques)
        self.lp_hist.append(self.loglik())

    def splitmerge_move(self):
        # pick a random edge randomising direction because otherwise
        # the distributions will be different in the split
        i, j = randchoice(self.links)
        sender, receiver = (i, j) if np.random.rand() < 0.5 else (j, i)
        
        # pick two random cliques containing them
        clique1 = randchoice(self.nodetocliques[sender])
        clique2 = randchoice(self.nodetocliques[receiver])

        if clique1 != clique2: # propose merge
            # check if merge is valid otherwise exit program
            cliqueset1 = self.cliquetonodes[clique1] 
            cliqueset2 = self.cliquetonodes[clique2] 
            for i in cliqueset1:
                for j in cliqueset2:
                    # check if it would introduce a new clique
                    if (i != j) and (ord(i, j) not in self.linktocliques): # is it a valid new link?
                        return # exit program

            # note: proposed clique = union
            union = cliqueset1 | cliqueset2
            intersection = cliqueset1 & cliqueset1

            # calculate the backward probability 
            sender_numcliques = len(self.nodetocliques[sender])
            receiver_numcliques = len(self.nodetocliques[receiver])
            sender_in_clique2 = int(sender in cliqueset2)
            receiver_in_clique1 = int(receiver in cliqueset1)

            lqsplit = - 2.0 * np.log(2.0) \
                - (len(union) - 2.0) * np.log(3.0) \
                - np.log(sender_numcliques - sender_in_clique2) \
                - np.log(receiver_numcliques - receiver_in_clique1)
            lqmerge = -np.log(sender_numcliques) - np.log(receiver_numcliques)

            lpsplit = 0.0
            for node in range(self.num_nodes):
                if node not in intersection:
                    lqsplit +=  np.log(self.num_cliques + self.c - 1.0 - self.nodecount[node] + self.sigma)
                else:
                    # <mauricio: check this one!!!> 
                    if self.nodecount[node] - self.sigma <= 0:
                        pdb.set_trace()
                    lqsplit +=  np.log(self.nodecount[node] - self.sigma) 
            lpsplit += - self.num_nodes * np.log(self.num_cliques + self.c - 1.0)

            lpmerge = np.log(self.num_cliques / self.lamb)
            
            newstuff = [node for node in intersection if self.nodecount[node] == 2] \
                + [node for node in (union - intersection) if self.nodecount[node] == 1]
            
            if len(newstuff) > 0:
                # then we have things that only appear in the merged clique
                log_poiss_param_1 = np.log(self.alpha) + gammaln(1 + self.c) \
                    + gammaln(self.num_cliques - 2 + self.c + self.sigma) \
                    - gammaln(self.num_cliques-1+self.c) - gammaln(self.c + self.sigma)
                log_poiss_param_2 = np.log(self.alpha) + gammaln(1 + self.c) \
                    + gammaln(self.num_cliques - 1 + self.c + self.sigma) \
                    - gammaln(self.num_cliques + self.c) - gammaln(self.c + self.sigma)

                nn_merge = len(newstuff)
                nn_split_1 = len(set(newstuff) & cliqueset1)
                nn_split_2 = nn_merge - nn_split_1

                lpmerge += nn_merge*log_poiss_param_1 - np.exp(log_poiss_param_1) - gammaln(nn_merge + 1)
                lpsplit += nn_split_1*log_poiss_param_1 - np.exp(log_poiss_param_1) - gammaln(nn_split_1 + 1)
                lpsplit += nn_split_2*log_poiss_param_2 - np.exp(log_poiss_param_2) - gammaln(nn_split_2 + 1)
            else:
                log_poiss_param = np.log(self.alpha) + gammaln(1+self.c) \
                    + gammaln(self.num_cliques - 1 +self.c + self.sigma) \
                    - gammaln(self.num_cliques+self.c) - gammaln(self.c + self.sigma)
                lpsplit += - np.exp(log_poiss_param)

            laccept = lpmerge - lpsplit + lqsplit - lqmerge

            if np.log(np.random.rand()) < laccept: 
                # update nodetocliques / nodecount
                for node in cliqueset2:
                    if node in cliqueset1:
                        self.nodecount[node] -= 1
                    self.nodetocliques[node].add(clique1)
                    self.nodetocliques[node].remove(clique2)
                
                # update linktocliques
                for i in cliqueset2:
                    for j in cliqueset2:
                        if i < j:
                            self.linktocliques[(i, j)].remove(clique2)
                            self.linktocliques[(i, j)].add(clique1)
                for i in cliqueset1:
                    for j in cliqueset2 - intersection:
                        self.linktocliques[ord(i, j)].add(clique1)
                        
                # update cliquetonodes
                self.cliquetonodes[clique1] = union
                self.cliquetonodes[clique2] = set()

                # update number of cliques
                self.num_cliques -= 1

        else:  # clique1=clique2 propose a split

            # propose a split
            split1 = {sender}
            split2 = {receiver}
            if np.random.rand() < 0.5:
                split1.add(receiver)
            if np.random.rand() < 0.5:
                split2.add(sender)

            # for the others do an unbalanced split
            cliqueset = self.cliquetonodes[clique1]
            for node in cliqueset:
                if np.random.rand() < 2/3:
                    split1.add(node)
                else:
                    split2.add(node)
            intersection = split1 & split2

            # check if split is valid, otherwise exit program
            if len(split1) < 2 or len(split2) < 2:
                return # bad proposal --> not sure what we should do here, we should propose again probably or marginal mcmc won't be correct
            # list links will be in the difference set of the splits
            for i in split1 - intersection:
                for j in split2 - intersection:
                    # check if only the current clique has this link
                    if len(self.linktocliques[ord(i, j)]) == 1:
                        return # exit program
            
            #all the denominators are the same...
            lpsplit = -self.num_nodes * np.log(self.num_cliques + self.c)
            for node in range(self.num_nodes):
                if node not in cliqueset:
                    lpsplit += np.log(self.num_cliques + self.c - self.nodecount[node] + self.sigma)
                elif node in split1 and node in split2: # actually only sender or receiver can...
                    lpsplit += np.log(self.nodecount[node] + 1 - self.sigma)
                else:
                    lpsplit += np.log(self.num_cliques + self.c - self.nodecount[node] - 1 + self.sigma)

            newstuff = [node for node in cliqueset if self.nodecount[node] == 1]
            if len(newstuff) > 0:
                #then we have things that appear for the first time in clique_i
                log_poiss_param_1 = np.log(self.alpha) + gammaln(1 + self.c) \
                    + gammaln(self.num_cliques - 2 + self.c + self.sigma) \
                    - gammaln(self.num_cliques - 1 + self.c) - gammaln(self.c + self.sigma)
                log_poiss_param_2 = np.log(self.alpha) + gammaln(1 + self.c) \
                    + gammaln(self.num_cliques - 1 + self.c + self.sigma) \
                    - gammaln(self.num_cliques+self.c) - gammaln(self.c + self.sigma)

                nn_merge = len(newstuff)
                nn_split_1 = len(set(newstuff) & self.cliquetonodes[clique1])
                nn_split_2 = nn_merge - nn_split_1

                log_poiss_param_1 = np.log(self.alpha) + gammaln(1 + self.c) \
                    + gammaln(self.num_cliques - 1 + self.c + self.sigma) \
                    - gammaln(self.num_cliques + self.c) - gammaln(self.c + self.sigma)
                log_poiss_param_2 = np.log(self.alpha) + gammaln(1+self.c) \
                    + gammaln(self.num_cliques+self.c + self.sigma) \
                    - gammaln(self.num_cliques+1+self.c) - gammaln(self.c + self.sigma)

                nn_merge = len(newstuff)
                nn_split_1 = len(split1)
                nn_split_2 = len(split2)

                lpmerge = nn_merge*log_poiss_param_1 - np.exp(log_poiss_param_1) - gammaln(nn_merge + 1)
                lpsplit += nn_split_1*log_poiss_param_1 - np.exp(log_poiss_param_1) - gammaln(nn_split_1 + 1)
                lpsplit += nn_split_2*log_poiss_param_2 - np.exp(log_poiss_param_2) - gammaln(nn_split_2 + 1)
                lpsplit -= lpmerge
                
            else:
                # the Kth row is the same for both, then add 0 to the K+1th
                log_poiss_param = np.log(self.alpha) + gammaln(1+self.c) \
                    + gammaln(self.num_cliques + self.c + self.sigma) - gammaln(self.num_cliques + 1 + self.c) \
                    - gammaln(self.c + self.sigma)
                lpsplit += -np.exp(log_poiss_param) \
                    + np.log(self.lamb / (self.num_cliques + 1))
                

            sender_numcliques = len(self.nodetocliques[sender])
            receiver_numcliques = len(self.nodetocliques[receiver])
            sender_in_clique2 = int(sender in self.cliquetonodes[clique2])
            receiver_in_clique1 = int(receiver in self.cliquetonodes[clique1])
            
            lqsplit = -2.0 * np.log(2.0) \
                - (len(cliqueset) - 2.0) * np.log(3.0) \
                - np.log(sender_numcliques) \
                - np.log(receiver_numcliques)

            lqmerge = -np.log(self.nodecount[sender] + sender_in_clique2) \
                - np.log(self.nodecount[receiver] + receiver_in_clique1)

            laccept = lpsplit - lqsplit + lqmerge

            if np.log(np.random.rand()) < laccept:
                # update nodetocliques / nodecount
                newclique = len(self.cliquetonodes)
                for node in split2 - intersection: # the ones in new split only, remove pointer to previous node
                    self.nodetocliques[node].add(newclique)
                    self.nodetocliques[node].remove(clique1)
                
                for node in intersection: # te ones in intersection just add new pointer
                    self.nodetocliques[node].add(newclique)
                    self.nodecount[node] += 1

                # update linktocliques
                for i in split1:
                    for j in split2 - intersection:
                            self.linktocliques[ord(i, j)].remove(clique1)
                for i in split2:
                    for j in split2:
                        if i < j:
                            self.linktocliques[(i, j)].add(newclique)

                # update cliquetonodes
                self.cliquetonodes[clique1] = split1
                self.cliquetonodes.append(split2)

                # update numberofcliques
                self.num_cliques += 1

    def splitmerge(self, N=1):
        for _ in range(N):
            self.splitmerge_move()

        # tidy up
        self.cliquetonodes = [x for x in self.cliquetonodes if len(x) > 0]
        self.nodetocliques = init_nodetocliques(self.cliquetonodes, self.num_nodes)
        self.linktocliques = init_linktocliques(self.cliquetonodes, self.links)

    def loglik(self, alpha=None, sigma=None, c=None):
        alpha = alpha if alpha is not None else self.alpha
        sigma = sigma if sigma is not None else self.sigma
        c = c if c is not None else self.c

        ll = -poissonparams(self.num_cliques, alpha, sigma, c).sum() \
            - self.num_nodes * gammaln(1.0 - sigma) + \
            - self.num_nodes * gammaln(c + sigma) \
            - self.num_nodes * gammaln(self.num_cliques + c) \
            + self.num_nodes * gammaln(1.0 + c) \
            + gammaln(self.nodecount - sigma).sum() \
            + gammaln(self.num_cliques - self.nodecount + c + sigma).sum() \
            + self.num_nodes * np.log(alpha)

        return ll

    def loglik_grad(self, h=1e-6):
        a, s, c = self.alpha, self.sigma, self.c

        # gradient of alpha is easy to obtain analytically: TODO!
        # const = self.poissonparams(alpha=1.0).sum()
        # alpha_grad = self.num_nodes / a - const

        # other gradients numerically
        alphagrad = 0.5 * (self.loglik(alpha=a + h) - self.loglik(alpha=a - h)) / h
        sigmagrad = 0.5 * (self.loglik(sigma=s + h) - self.loglik(sigma=s - h)) / h
        cgrad = 0.5 * (self.loglik(c=c + h) - self.loglik(c=c - h)) / h

        return alphagrad, sigmagrad, cgrad
    
    def optim(self, num_iters=1, stepsize=0.01, momentum=0.3, momentum_free_iters=0, backtracktol=0.001):
        # init grad vars
        dalpha, dsigma, dc = 0.0, 0.0, 0.0
        alphagrad, sigmagrad, cgrad, stepsize = self.backtracksearch_gradient(currstepsize=stepsize, tol = backtracktol)
    
        for i in range(num_iters):
            # update variables
            m = momentum if i > momentum_free_iters else 0.
            dalpha = m * dalpha + stepsize * alphagrad
            dsigma = m * dsigma + stepsize * sigmagrad
            dc = m * dc + stepsize * cgrad
            
            # must recheck domain due to momentum
            self.alpha = max(self.alpha + dalpha, 0.001)
            self.sigma = np.clip(self.sigma + dsigma, 0.001, 0.999)
            self.c = max(self.c + dc, -self.sigma + 0.001)
            
            # gradients again
            alphagrad, sigmagrad, cgrad, stepsize = \
                self.backtracksearch_gradient(currstepsize = stepsize, tol = backtracktol)

    def backtracksearch_gradient(self, currstepsize = 0.01, maxiters=100, tol=0.001):
        currentll = self.loglik()
        alphagrad, sigmagrad, cgrad = self.loglik_grad()

        # do alpha separately first
        i = 0
        stepsize = 2.0 * currstepsize # start inflated
        while i < maxiters:
            # deal with alpha step size separetely
            alphanew = self.alpha + stepsize * alphagrad
            sigmanew = self.sigma + stepsize * sigmagrad
            cnew = self.c + stepsize * cgrad

            if alphanew > 0 and 0 < sigmanew < 1 and cnew > -sigmanew: # then make step smaller
                # check if new point is acceptable
                llnew = self.loglik(alpha=alphanew, sigma=sigmanew, c=cnew) 
                if llnew > currentll - tol:
                    # print("stepsize: {}     new: {}     old: {}   tol {}".format(stepsize, llnew, currentll, tol))
                    break
            
            stepsize *= 0.5
            i += 1
        
        
        return alphagrad, sigmagrad, cgrad, stepsize


    def samplelamb(self, step_size = 0.01):
        lamb_a = self.lambda_params[0] + self.num_cliques
        lamb_b = self.lambda_params[1] + 1.
        self.lamb = np.random.gamma(lamb_a, 1.0 / lamb_b)


def sample_from_ibp(K, alpha, sigma, c):
    """
    samples from the random clique cover model using the three parameter ibp
    params
        K: number of random cliques
        alpha, sigma, c: ibp parameters
    returns 
        a sparse matrix, compressed by rows, representing the clique membership matrix
        recover the adjacency matrix with min(Z'Z, 1)
    """
    pp = poissonparams(K, alpha, sigma, c)
    new_nodes = np.random.poisson(pp)
    Ncols = new_nodes.sum()
    node_count = np.zeros(Ncols)

    # used to build sparse matrix, entries of each Zij=1
    colidx = [] 
    rowidx = []
    rightmost_node = 0

    # for each clique
    for n in range(K):
        # revisit each previously seen node
        for k in range(rightmost_node):
            prob_repeat = (node_count[k] - sigma) / (n + c)
            r = np.random.rand()
            if r < prob_repeat:
                rowidx.append(n)
                colidx.append(k)
                node_count[k] += 1

        for k in range(rightmost_node, rightmost_node + new_nodes[n]):
            rowidx.append(n)
            colidx.append(k)
            node_count[k] += 1
        
        rightmost_node += new_nodes[n]

    # build sparse matrix
    data = np.ones(len(rowidx), int)
    shape = (K, Ncols)
    Z = csr_matrix((data, (rowidx, colidx)), shape)

    return Z
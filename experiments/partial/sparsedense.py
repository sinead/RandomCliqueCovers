import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.stats as ss
#from scipy.stats import poisson,gamma
from scipy.special import gammaln, gamma
import networkx as nx
#import cProfile
import pickle

class NetworkFull(object):
    def __init__(self, network,links,alpha, sigma,lamb, c):
        self.alpha = alpha
        self.sigma = sigma
        self.lamb = lamb
        self.c = c
        self.network = network
        self.links = links
        self.num_nodes = network.shape[0]
        self.num_links = len(links)
        self.Z = np.zeros((self.num_links, self.num_nodes))
        for i in range(self.num_links):
            self.Z[i,self.links[i][0]]=1
            self.Z[i,self.links[i][1]]=1
            
        self.K = self.Z.shape[0]
        self.alpha_params = [1.,1.]
        self.sigma_params = [1.,1.]
        self.c_params = [1., 1.]
        self.lambda_params = [1.,1.]

        self.Z_hist = []
        self.alpha_hist = []
        self.sigma_hist = []
        self.c_hist = []
        self.lambda_hist = []
        self.K_hist = []
        self.lp_hist = []
        

    def add_mask(self,mask=None, hold_out_percent = None):
        if mask is not None:
            self.mask=mask
        else:
            mask_prob = np.random.rand(self.network.shape)
            self.mask = (mask_prob<hold_out_percent).astype(int)
            self.mask = np.triu(self.mask,1)
            self.mask = self.mask + self.mask.T
        self.recovered_net = np.minimum(np.dot(self.Z.T,self.Z), 1)
        np.fill_diagonal(self.recovered_net,0)
            
    def save_sample(self,lp, write_Z = False):
        if write_Z:
            self.Z_hist.append(self.Z)
        self.alpha_hist.append(self.alpha)
        self.sigma_hist.append(self.sigma)
        self.c_hist.append(self.c)
        self.lambda_hist.append(self.lamb)
        self.K_hist.append(self.K)
        self.lp_hist.append(lp)

    def write_samples(self,filename,write_Z = False):
        if write_Z:
            current = {'Zs': self.Z_hist, 'alphas' : self.alpha_hist, 'sigmas' : self.sigma_hist, 'cs' : self.c_hist, 'lambdas' : self.lambda_hist, 'Ks' : self.K_hist, 'lps' : self.lp_hist}
        else:
            current = {'alphas' : self.alpha_hist, 'sigmas' : self.sigma_hist, 'cs' : self.c_hist, 'lambdas' : self.lambda_hist, 'Ks' : self.K_hist, 'lps' : self.lp_hist}
        with open(filename,"wb") as f:
            pickle.dump(current,f)

    def init_from_file(self,filename):
        with open(filename, "rb") as f:
            last_state = pickle.load(f)

        self.Z_hist = last_state['Zs']
        self.alpha_hist = last_state['alphas']
        self.sigma_hist = last_state['sigmas']
        self.c_hist = last_state['cs']
        self.lambda_hist = last_state['lambdas']
        self.K_hist = last_state['Ks']
        self.lp_hist = last_state['lps']

        self.Z = self.Z_hist[-1]
        self.K = self.K_hist[-1]
        self.sigma = self.sigma_hist[-1]
        self.alpha = self.alpha_hist[-1]
        self.c = self.c_hist[-1]
        self.lamb = self.lambda_hist[-1]
        

    def remove_singletons(self,min_count=2):
        vertex_counts = np.sum(self.network,0)
        singletons = np.where(vertex_counts<min_count)[0]
        self.network = np.delete(self.network, singletons,0)
        self.network = np.delete(self.network, singletons, 1)
        vertex_counts = np.sum(self.network,0)
        if np.any(vertex_counts==0):
            self.network = np.delete(self.network, np.where(vertex_counts==0)[0],0)
            self.network = np.delete(self.network, np.where(vertex_counts==0)[0],1)
        self.num_links = np.sum(np.triu(self.network,1)).astype(int)
        self.links = np.array(np.where(np.triu(self.network,1)==1)).T

        self.num_nodes = self.network.shape[0]
        self.Z = np.zeros((self.num_links, self.num_nodes))
        for i in range(self.num_links):
            self.Z[i,self.links[i][0]]=1
            self.Z[i,self.links[i][1]]=1
            
        self.K = self.Z.shape[0]

        
    def set_Z(self,Z):
        self.Z = Z
        self.K = self.Z.shape[0]
        
    def clique_init(self):
        G = nx.Graph()
        G.add_edges_from(self.links)
        all_cliques = list(nx.find_cliques(G))
        self.K = len(all_cliques)
        self.Z = np.zeros((self.K,self.num_nodes))
        for k in range(self.K):
            for n in range(len(all_cliques[k])):
                self.Z[k,all_cliques[k][n]]=1
            
    def set_hypers(self, alpha_params=None, sigma_params = None, c_params = None, lambda_params = None):
        if alpha_params is not None:
            self.alpha_params = alpha_params
        if sigma_params is not None:
            self.sigma_params = sigma_params
        if c_params is not None:
            self.c_params = c_params
        if lambda_params is not None:
            self.lambda_params = lambda_params
                

    
    def sample(self, num_iters = 1000, num_sm = 10,num_hyper = 10,save_every=100,write_every=1000,sample_hypers=True,hyper_start = 0,do_gibbs = False, burnin=0,verbose=True,filename = 'sd_hist.pkl'):
        #I don't find Gibbs helps much for the fully observed model, and it's slow.
        for iter in range(num_iters):
            if do_gibbs:
                self.gibbs()
            
            
            for sm in range(num_sm):
                self.splitmerge()
                
            if sample_hypers:
                if iter > hyper_start:
                    for sh in range(num_hyper):
                        self.sample_hypers()
                    
            if iter%save_every==0:
                lp=self.log_joint()
                if iter>burnin:
                    self.save_sample(lp)
                if verbose:
                    # ADD print c
                    print('iter ',iter,', K=',self.K,', alpha=',self.alpha, ', sigma=',self.sigma, 'c=', self.c,', lp=',lp)
            if iter>burnin:
                if iter%write_every==0:
                    if verbose:
                        print('writing to file...')
                    self.write_samples(filename)
                    if verbose:
                        print('done!')


    def plot_log_lik_alpha(self,alpha_min, alpha_max,sigma=None):
        if sigma is None:
            sigma=self.sigma
        xvals = np.linspace(alpha_min, alpha_max,num=100)
        yvals = np.zeros(100)
        for i in range(100):
            yvals[i] = self.log_lik(alpha=xvals[i] , alpha_only = True)

        return xvals,yvals

    def log_joint(self):

        # ADD lp_sigma
        lp_alpha = ss.gamma.logpdf(self.alpha,self.alpha_params[0],scale = 1/self.alpha_params[1])
        lp_sigma = ss.beta.logpdf(self.sigma,self.sigma_params[0],self.sigma_params[1])
        
        #let's put the prior on c, as a gamma prior on c+sigma

        c_diff = self.c + self.sigma

        lp_c = ss.gamma.logpdf(c_diff, self.c_params[0], scale = 1/self.c_params[1])

        lp_lambda = ss.gamma.logpdf(self.lamb, self.lambda_params[0], scale = 1/self.lambda_params[1])
        
        ll_Z = self.log_lik(include_K = True)

        lp = lp_alpha + lp_sigma + lp_c + lp_lambda +  ll_Z
        return lp

        
    def log_lik(self,sigma=None, c = None, alpha=None, alpha_only = False, include_K = False):
        #change to include c. ADD
        # this is based on Teh et al eq 10
        
        if sigma is None:
            sigma = self.sigma
        if alpha is None:
            alpha = self.alpha
        if c is None:
            c = self.c
        ll = self.num_nodes*np.log(alpha)
        const_lngamma_ratio = gammaln(c + 1) - gammaln(c + sigma)
        for i in range(1, self.K+1):
            var_lngamma_ratio = gammaln(i - 1 + c + sigma) - gammaln(i + c)
            ll -= alpha*np.exp(const_lngamma_ratio + var_lngamma_ratio)

        if alpha_only is False:
            mk = np.sum(self.Z, 0)
            ll += self.num_nodes*(gammaln(1+c) - gammaln(1-sigma) - gammaln(c+sigma) - gammaln(self.K+c))

            ll += np.sum(gammaln(mk - sigma))
            ll += np.sum(gammaln(self.K - mk + c + sigma))

        if include_K:
            ll+=ss.poisson.logpmf(self.K, self.lamb)
        if np.isinf(ll):
            pdb.set_trace() #shouldn't happen
                         
        return ll
                              
            
    
    def sample_hypers(self,step_size = 0.01):
        # ADD sample c (note: c>-sigma
        mk = np.sum(self.Z,0)
        alpha_prop = self.alpha+10*step_size*np.random.randn()
        if alpha_prop>0:
            lp_ratio = (self.alpha_params[0]-1)*(np.log(alpha_prop)-np.log(self.alpha)) + self.alpha_params[1]*(self.alpha-alpha_prop)

            ll_new = self.log_lik(alpha = alpha_prop,alpha_only = True)
            ll_old = self.log_lik(alpha_only = True)
            lratio = ll_new - ll_old + lp_ratio
            r = np.log(np.random.rand())

            if r<lratio:
                self.alpha = alpha_prop
        sigma_prop = self.sigma+step_size*np.random.randn()
        if sigma_prop>0:
            if sigma_prop<1:
                # we need c>-sigma 
                if sigma_prop > -1*self.c:
                    ll_new = self.log_lik(sigma = sigma_prop)
                    c_diff_new = self.c + sigma_prop

                    ll_new += ss.gamma.logpdf(c_diff_new, self.c_params[0], scale = 1/self.c_params[1])
                    
                    ll_old = self.log_lik()
                    c_diff_old = self.c + self.sigma

                    ll_old += ss.gamma.logpdf(c_diff_old, self.c_params[0], scale = 1/self.c_params[1])
                
                    lp_ratio = (self.sigma_params[0]-1)*(np.log(sigma_prop)-np.log(self.sigma)) + (self.sigma_params[1]-1)*(np.log(1-sigma_prop)-np.log(1-self.sigma))
                    lratio = ll_new - ll_old + lp_ratio
                    r = np.log(np.random.rand())
                
                    if r<lratio:
                        self.sigma = sigma_prop

        c_prop = self.c + step_size*np.random.randn()
        if c_prop > -1*self.sigma:
            ll_new = self.log_lik(c=c_prop)
            c_diff_new = c_prop + self.sigma
            lp_new = ss.gamma.logpdf(c_diff_new, self.c_params[0], scale = 1/self.c_params[1])

            ll_old = self.log_lik()
            c_diff_old = self.c + self.sigma
            lp_old = ss.gamma.logpdf(c_diff_old, self.c_params[0], scale = 1/self.c_params[1])

            lratio = ll_new - ll_old + lp_new - lp_old
            r = np.log(np.random.rand())
            if r<lratio:
                self.c = c_prop

        lamb_a = self.lambda_params[0] + self.K
        lamb_b = self.lambda_params[1] + 1.
        self.lamb = np.random.gamma(lamb_a, 1/lamb_b)
   
    def gibbs(self):
        empty_cliques = []
        mk = np.sum(self.Z,0)
        for node in range(self.num_nodes):
            cic = np.dot(self.Z[:,node].T,self.Z)
            for clique in range(self.K):
                if self.Z[clique,node]==1:
                    #can we remove it?
                    cic_rem = cic - self.Z[clique,:]
                    uncovered = (1-np.minimum(cic_rem,1))*self.network[node,:]
                    if np.sum(uncovered)==0:
                        mk[node]-=1
                        if mk[node]==0:
                            pdb.set_trace() #shouldn't be possible, because we should break the uncovered check
                        #if it doesn't affect the network... sample from the prior
                        p0 = (self.K-mk[node])/(self.K-self.sigma)
                        
                        r = np.random.rand()
                        if r<p0:
                            self.Z[clique,node]=0
                            cic_tmp = np.dot(self.Z[:,node].T, self.Z)
                            if np.any(cic_tmp!=cic_rem):
                                pdb.set_trace()
                            cic = cic_rem+0
                            
                        else:
                            mk[node]+=1
                else:
                    #can we add it?
                    cic_plus = cic + self.Z[clique,:]
                    cic_plus = np.minimum(cic_plus,1)
                    overcovered = cic_plus*(1-self.network)
                    overcovered[node]=0
                    if np.sum(overcovered)==0:
                        p1 = (mk[node]-self.sigma)/(self.K + self.c)
                        r = np.random.rand()
                        if r<p1:
                            self.Z[clique,node]=1
                            cic = cic+self.Z[clique,:]
                            mk[node]+=1


    def gibbs_mask(self):
        empty_cliques = []
        mk = np.sum(self.Z,0)
        for node in range(self.num_nodes):
            cic = np.dot(self.Z[:,node].T,self.Z)
            for clique in range(self.K):
                if self.Z[clique,node]==1:
                    #can we remove it?
                    cic_rem = cic - self.Z[clique,:]
                    uncovered = (1-np.minimum(cic_rem,1))*self.network[node,:]*(1-self.mask[node,:])
                    if np.sum(uncovered)==0:
                        mk[node]-=1
                        if mk[node]==0:
                            pdb.set_trace() #shouldn't be possible, because we should break the uncovered check
                        #if it doesn't affect the network... sample from the prior
                        p0 = (self.K-mk[node])/(self.K+self.c)
                        
                        r = np.random.rand()
                        if r<p0:
                            self.Z[clique,node]=0
                            cic = cic_rem+0
                            
                        else:
                            mk[node]+=1
                else:
                    #can we add it?
                    cic_plus = cic + self.Z[clique,:]
                    cic_plus = np.minimum(cic_plus,1)
                    overcovered = cic_plus*(1-self.network[node,:])*(1-self.mask[node,:])
                    overcovered[node]=0
                    if np.sum(overcovered)==0:
                        p1 = (mk[node]-self.sigma)/(self.K+self.c)
                        r = np.random.rand()
                        if r<p1:
                            self.Z[clique,node]=1
                            cic = cic+self.Z[clique,:]
                            mk[node]+=1



    def splitmerge(self):

        #pick an edge
        link_id = np.random.choice(self.num_links)
        r = np.random.rand()
        if r<0.5:
            sender = self.links[link_id][0]
            receiver = self.links[link_id][1]
        else:
            #randomizing because otherwise the distributions will be different in the split
            sender = self.links[link_id][1]
            receiver = self.links[link_id][0]
        #pick the first clique
        valid_cliques = np.where(self.Z[:,sender]==1)[0]
        clique_i = valid_cliques[np.random.choice(len(valid_cliques))]
        valid_cliques = np.where(self.Z[:,receiver]==1)[0]
        clique_j = valid_cliques[np.random.choice(len(valid_cliques))]
        if clique_i == clique_j:
            #propose split

            Z_add = np.zeros((2,self.num_nodes))

            lqsplit = 0
            lpsplit = 0
            
            mk = np.sum(self.Z,0) -self.Z[clique_i,:]

            num_ones = int(np.sum(self.Z[clique_i,:]))
            Z_tmp = np.zeros((2,num_ones))
            r = np.random.rand(num_ones)
            Z_tmp[0,r<(2/3)] = 1
            Z_tmp[1,r>(1/3)] = 1
            Z_add[:,np.where(self.Z[clique_i,:]==1)[0]] = Z_tmp
            #deal with sender and receiver

            Z_add[0,sender]=1
            Z_add[1,receiver]=1
            r = np.random.rand()
            if r<0.5:
                Z_add[1,sender]=1
            else:
                Z_add[1,sender]=0

            r = np.random.rand()
            if r<0.5:
                Z_add[0,receiver]=1
            else:
                Z_add[0,receiver]=0

            


            mktmp = np.sum(Z_add,0)


            
            # lpsplit should have the following additional prob over merged:
            # mktmp[k] = 0 (i.e. was zero, added a zero):
            # an extra log(p(Z_{n+1, k} = 0|z_{n,k} = 0, mk[k]) = log((n+c-mk[k]+sigma)/(n+c))
            #
            # mktmp[k] = 1 (i.e. was one, added a zero):
            #an estra log(p(Z_{n+1, k} = 0|Z_{n,k} = 1, mk[k]) = log(n+c-mk[k]-1+Sigma)/(n+c))
            
            # mktmp[k] = 2 (i.e. was one, added a one):
            # and extra log(p(Z_{n+1,k} = 1| z_{n,k} = 1, mk[k]) = log((mk[k]+1-sigma)/(n+c))

            #all the denominators are the same...
            lpsplit = -self.num_nodes*np.log(self.K+self.c)

            # mktmp[k] = 0
            lpsplit += np.sum(np.log(self.K + self.c - mk[mktmp==0] + self.sigma))
            #mktmp[k] = 1
            lpsplit += np.sum(np.log(self.K + self.c - mk[mktmp==1] - 1 +self.sigma))

            #mktmp[k] = 2
            lpsplit += np.sum(np.log(mk[mktmp==2] + 1 - self.sigma))

  

            if np.any(mk==0):
                #then we have things that appear for the first time in clique_i


                # Number of new things is Poisson(f(alpha)), where f(alpha) is from p1 of Teh et al
               
                log_poiss_param_1 = np.log(self.alpha) + gammaln(1+self.c) + gammaln(self.K - 1 + self.c + self.sigma) - gammaln(self.K + self.c) - gammaln(self.c + self.sigma)
                log_poiss_param_2 = np.log(self.alpha) + gammaln(1+self.c) + gammaln(self.K+self.c + self.sigma) - gammaln(self.K+1+self.c) - gammaln(self.c + self.sigma)

                nn_merge = np.sum(mk==0)
                lpmerge = nn_merge*log_poiss_param_1 -np.exp(log_poiss_param_1) - gammaln(nn_merge+1)
                nn_split_1 = np.sum(Z_add[0,mk==0])
                nn_split_2 = nn_merge-nn_split_1

                lpsplit += nn_split_1*log_poiss_param_1 - np.exp(log_poiss_param_1) - gammaln(nn_split_1+1)
                lpsplit += nn_split_2*log_poiss_param_2 - np.exp(log_poiss_param_2) - gammaln(nn_split_2+1)
                lpsplit -=lpmerge
                
            else:
                # the Kth row is the same for both, then add 0 to the K+1th
                log_poiss_param = np.log(self.alpha) + gammaln(1+self.c) + gammaln(self.K+self.c+self.sigma) - gammaln(self.K+1+self.c) - gammaln(self.c+self.sigma)
                lpsplit += -np.exp(log_poiss_param)
                
            lqsplit = -2*np.log(2) - (num_ones-2)*np.log(3)
            
            
            #is the resulting proposal valid?
            cic = np.dot(self.Z.T,self.Z)
            cic_prop = cic - np.outer(self.Z[clique_i,:],self.Z[clique_i,:]) + np.dot(Z_add.T,Z_add)
            cic = np.minimum(cic,1)
            cic_prop = np.minimum(cic_prop,1)
            np.fill_diagonal(cic,0)
            np.fill_diagonal(cic_prop,0)
            
            if np.all(cic_prop==cic): #if it's consistent with the network
               
                #then calculate the acceptance prob
                lqsplit = lqsplit -np.log(np.sum(self.Z[:,sender]))-np.log(np.sum(self.Z[:,receiver]))
                lqmerge = -np.log(mk[sender]+np.sum(Z_add[:,sender])) - np.log(mk[receiver]+np.sum(Z_add[:,receiver]))

                
                lpsplit += np.log(self.lamb/(self.K+1))
                laccept = lpsplit -lqsplit+ lqmerge
                r = np.log(np.random.rand())
                
                if r<laccept:

                    self.Z[clique_i,:] = Z_add[0,:]+0
                    #self.Z = np.delete(self.Z,clique_i,0)
                    self.Z = np.vstack((self.Z,Z_add[1,:]))
                    #self.Z = Z_prop + 0
                    self.K+=1

        else:
            #propose merge
            Z_sum = self.Z[clique_i,:]+self.Z[clique_j,:]
            Z_prop = np.minimum(Z_sum,1)
            
            ZZ = np.outer(Z_prop,Z_prop)      
            check_ZZ = (1-self.network)*ZZ
            np.fill_diagonal(check_ZZ,0)
            if np.sum(check_ZZ)==0:
                #merge OK, proceed
                mk = np.sum(self.Z,0)-Z_sum 
                #calculate the backward probability
                num_affected = np.sum(Z_prop)
                if num_affected<2:
                    pdb.set_trace() #shouldn't happen
                num_ones = int(np.sum(Z_prop))
                lqsplit = -2*np.log(2) - (num_ones-2)*np.log(3)
                lqsplit = lqsplit - np.log(np.sum(self.Z[:,sender])-self.Z[clique_i,sender]-self.Z[clique_j,sender]+1)
                lqsplit = lqsplit- np.log(np.sum(self.Z[:,receiver])-self.Z[clique_i,receiver]-self.Z[clique_j,receiver]+1)

                lqmerge = -np.log(np.sum(self.Z[:,sender])) - np.log(np.sum(self.Z[:,receiver]))
                
                lpsplit =0
                for node in range(self.num_nodes):
                    if Z_sum[node]==0:
                        #mk is the same, and K the same, p0
                        lpsplit = lpsplit + np.log(self.K + self.c - 1 -mk[node] + self.sigma) - np.log(self.K+ self.c - 1)
                    elif Z_sum[node]==1:
                        #mk is plus one, and K the same, p0
                        lpsplit = lpsplit + np.log(self.K + self.c -2 -mk[node]) - np.log(self.K + self.c -1)
                    else:
                        #mk is plus one, and K the same, p2
                        lpsplit = lpsplit + np.log(mk[node]+1-self.sigma) - np.log(self.K+self.c - 1)
                        
                lpmerge = np.log(self.K/self.lamb)
                if np.any(mk==0):
                    #then we have things that appear for the first time in clique_i
                    log_poiss_param_1 = np.log(self.alpha) + gammaln(1+self.c) + gammaln(self.K-2 + self.c + self.sigma) - gammaln(self.K-1+self.c) - gammaln(self.c + self.sigma)
                    log_poiss_param_2 = np.log(self.alpha) + gammaln(1+self.c) + gammaln(self.K -1 +self.c + self.sigma) - gammaln(self.K+self.c)- gammaln(self.c + self.sigma)

                    nn_merge = np.sum(mk==0)
                    lpmerge += (nn_merge*log_poiss_param_1 -np.exp(log_poiss_param_1) - gammaln(nn_merge+1))
                    nn_split_1 = np.sum(self.Z[clique_i,mk==0])
                    
                    nn_split_2 = nn_merge-nn_split_1

                    lpsplit += nn_split_1*log_poiss_param_1 - np.exp(log_poiss_param_1) - gammaln(nn_split_1+1)
                    lpsplit += nn_split_2*log_poiss_param_2 - np.exp(log_poiss_param_2) - gammaln(nn_split_2+1)
                
                
                else:
                    log_poiss_param = np.log(self.alpha) + gammaln(1+self.c) + gammaln(self.K - 1 +self.c + self.sigma) - gammaln(self.K+self.c) - gammaln(self.c + self.sigma)
                    lpsplit += -np.exp(log_poiss_param)
                
                
                laccept = lpmerge -lpsplit +lqsplit - lqmerge
                r = np.log(np.random.rand())
                
                if r<laccept:
                    
                    try:
                        self.Z[clique_i,:]=Z_prop+0
                    except ValueError:
                        pdb.set_trace()
                    self.Z = np.delete(self.Z,clique_j,0)
                    self.K-=1



#should probably also add c for the partial model.
#not done yet.

class NetworkPartial(object):
    def __init__(self, network, links, alpha, sigma,lamb, pie):
        self.alpha = alpha
        self.sigma = sigma
        self.lamb = lamb
        self.network = network - np.diag(np.diag(network))
        self.links = links
        self.num_nodes = network.shape[0]
        self.num_links = len(links)
        self.Z = np.zeros((self.num_links, self.num_nodes))
        for i in range(self.num_links):
            self.Z[i,self.links[i][0]]=1
            self.Z[i,self.links[i][1]]=1
            
        self.K = self.Z.shape[0]
        self.alpha_params = [1.,1.]
        self.sigma_params = [1.,1.]
        self.pie_params = [1.,1.]
        self.lambda_params = [10.,1.]
        self.pie = pie
        
    def set_hyperpriors(self,alpha_params = None, sigma_params = None, pie_params = None, lambda_params = None):
        if alpha_params is not None:
            self.alpha_params = alpha_params
        if sigma_params is not None:
            self.sigma_params = sigma_params
        if pie_params is not None:
            self.pie_params = pie_params
        if lambda_params is not None:
            self.lambda_params = lambda_params
            
    def sample(self, num_iters = 1000, num_sm = 10,dot_every=100,sample_hypers=True,do_gibbs = True, verbose=True):
        #Gibbs seems to matter here
        for iter in range(num_iters):
            if do_gibbs:
                self.gibbs()
            
            for sm in range(num_sm):
                self.splitmerge()
            if sample_hypers:
                self.sample_hypers()
            if verbose and iter%dot_every==0:
                print('iter ',iter,', K=',self.K)

                
    def log_lik(self,sigma=None, alpha=None, alpha_only = False):
        #same as full
        if sigma is None:
            sigma = self.sigma
        if alpha is None:
            alpha = self.alpha
        ll = self.num_nodes*np.log(alpha)
        if alpha_only is False:
            ll += self.num_nodes*(np.log(1-sigma) - gammaln(self.K+1-sigma))
        mk = np.sum(self.Z,0)
       
        c1 = gamma(2-sigma)*alpha
        for clique in range(1,self.K+1):
            log_gamma_ratio = gammaln(clique) - gammaln(clique+1-sigma)
            ll -= c1*np.exp(log_gamma_ratio)
        
        if alpha_only is False:
            for node in range(self.num_nodes):
                ll +=gammaln(mk[node]-sigma) + gammaln(self.K-mk[node]+1)
                
        if np.isinf(ll):
            pdb.set_trace()
        
        
        return ll
                              
            
    
    def sample_hypers(self,step_size = 0.01):
        #same as full, but with sampling pie added in
        mk = np.sum(self.Z,0)
        alpha_prop = self.alpha+step_size*np.random.randn()
        if alpha_prop>0:
            lp_ratio = (self.alpha_params[0]-1)*(np.log(alpha_prop)-np.log(self.alpha)) + self.alpha_params[1]*(self.alpha-alpha_prop)

            ll_new = self.log_lik(alpha = alpha_prop,alpha_only = True)
            ll_old = self.log_lik(alpha_only = True)
            lratio = ll_new - ll_old + lp_ratio
            r = np.log(np.random.rand())
            if r<lratio:
                self.alpha = alpha_prop
        sigma_prop = self.sigma+step_size*np.random.randn()
        if sigma_prop>0:
            if sigma_prop<1:
                ll_new = self.log_lik(sigma = sigma_prop)
                ll_old = self.log_lik()
                
                lp_ratio = (self.sigma_params[0]-1)*(np.log(sigma_prop)-np.log(self.sigma)) + (self.sigma_params[1]-1)*(np.log(1-sigma_prop)-np.log(1-self.sigma))
                lratio = ll_new - ll_old + lp_ratio
                r = np.log(np.random.rand())
                
                if r<lratio:
                    self.sigma = sigma_prop
                    
        pie_prop = self.pie + step_size*np.random.randn()
        if pie_prop>0:
            if pie_prop<1:
                ll_new = self.loglikZ(pie=pie_prop)
                ll_old = self.loglikZ()
                lp_ratio = (self.pie_params[0]-1)*(np.log(pie_prop)-np.log(self.pie))+(self.pie_params[1]-1)*(np.log(1-pie_prop)-np.log(1-self.pie))
                lratio = ll_new - ll_old + lp_ratio
                r = np.log(np.random.rand())
                if r<lratio:
                    self.pie = pie_prop
        lamb_a = self.lambda_params[0] + self.K
        lamb_b = self.lambda_params[1] + 1.
        self.lamb = np.random.gamma(lamb_a, 1/lamb_b)
                     
    
    def gibbs(self):
        empty_cliques = []
        mk = np.sum(self.Z,0)
        for node in range(self.num_nodes):
            for clique in range(self.K):
                if self.Z[clique,node]==1:
                    self.Z[clique,node] = 0
                    ll_0 = self.loglikZn(node)
                    self.Z[clique,node]=1
                    if not np.isinf(ll_0):
                        
                        ll_1 = self.loglikZn(node)
                        mk[node]-=1
                        if mk[node]==0:
                            pdb.set_trace() #shouldn't be possible, because we should break the ll check
                        #if it doesn't affect the network... sample from the prior
                        prior0 = (self.K - mk[node]) / (self.K - self.sigma)
                        
                        prior1 = 1-prior0
                        
                        lp0 = np.log(prior0) + ll_0
                        lp1 = np.log(prior1) +ll_1
                        lp0 = lp0 - np.logaddexp(lp0,lp1)
                        r = np.log(np.random.rand())
                        if r<lp0:
                            self.Z[clique,node]=0
                            
                        else:
                            mk[node]+=1
                else:
                    self.Z[clique,node]=1
                    ll_1 = self.loglikZn(node)
                    self.Z[clique,node]=0
                    if not np.isinf(ll_1):
                        ll_0 = self.loglikZn(node)
                       
                        #if it doesn't affect the network... sample from the prior
                        prior0 = (self.K-mk[node])/(self.K-self.sigma)
                        
                        prior1 = 1-prior0
                        
                        lp0 = np.log(prior0) + ll_0
                        lp1 = np.log(prior1) +ll_1
                        lp1 = lp1 - np.logaddexp(lp0,lp1)
                        r = np.log(np.random.rand())
                        if r<lp1:
                            self.Z[clique,node]=1
                            mk[node]+=1
                    
    def loglikZ(self,Z=None,pie=None):
        if Z is None:
            Z = self.Z
        if pie is None:
            pie = self.pie
        cic = np.dot(Z.T,Z)
        try:
            np.fill_diagonal(cic,0)#cic - np.diag(np.diag(cic))
        except ValueError:
            pdb.set_trace()
        #check whether cic is ever zero, when network is 1
        zero_check = (1-np.minimum(cic,1))*self.network
        if np.sum(zero_check)==0:
            p0 = (1-pie)**cic
            p1 = 1-p0
            network_mask = self.network+1
            network_mask = np.triu(network_mask,1)-1
            #network_mask = np.triu(self.network,1)
            lp = np.sum(np.log(p0[np.where(network_mask==0)])) + np.sum(np.log(p1[np.where(network_mask==1)]))
            
        else:
            lp = -np.inf
        return lp
        
    def loglikZn(self,node,Z=None):
        if Z is None:
            Z = self.Z
        cic = np.dot(Z[:,node].T,Z)
        cic[node] = 0
        #check whether cic is ever zero, when network is 1
        zero_check = (1-np.minimum(cic,1))*self.network[node,:]
        if np.sum(zero_check)==0:
            p0 = (1-self.pie)**cic
            p1 = 1-p0
            lp = np.sum(np.log(p0[np.where(self.network[node,:]==0)])) + np.sum(np.log(p1[np.where(self.network[node,:]==1)]))
            
        else:
            lp = -np.inf
        return lp    
        
        
    def splitmerge(self):
        #pick an edge
        link_id = np.random.choice(self.num_links)
        r = np.random.rand()
        if r<0.5:
            sender = self.links[link_id][0]
            receiver = self.links[link_id][1]
        else:
            #randomizing because otherwise the distributions will be different in the split
            sender = self.links[link_id][1]
            receiver = self.links[link_id][0]
        #pick the first clique
        valid_cliques = np.where(self.Z[:,sender]==1)[0]
        try:
            clique_i = valid_cliques[np.random.choice(len(valid_cliques))]
        except ValueError:
            pdb.set_trace()
        valid_cliques = np.where(self.Z[:,receiver]==1)[0]
        try:
            clique_j = valid_cliques[np.random.choice(len(valid_cliques))]
        except ValueError:
            pdb.set_trace()
        if clique_i == clique_j:
            #propose split
            Z_prop = self.Z+0
            Z_prop = np.delete(Z_prop,clique_i,0)
            Z_prop = np.vstack((Z_prop, np.zeros((2,self.num_nodes))))
            
            lqsplit = 0
            lpsplit = 0
            
            
            mk = np.sum(Z_prop,0) 
            for node in range(self.num_nodes):#np.random.permutation(self.num_nodes):
                if self.Z[clique_i,node]==1:
                    if node == sender:
                        #must be 11 or 10
                        Z_prop[self.K-1,node]=1
                        
                        r = np.random.rand()
                        if r<0.5:
                            Z_prop[self.K,node]=1
                            #mk is one bigger, and K is one bigger, p1
                            lpsplit =lpsplit + np.log(mk[node]+1-self.sigma) - np.log(self.K+1-self.sigma)
                        else:
                            #mk is one bigger, and K is one bigger, p0
                            lpsplit = lpsplit + np.log(self.K+1-mk[node]-1) - np.log(self.K+1-self.sigma)
                        lqsplit -=np.log(2)
                        
                    elif node==receiver:
                        #must be 11 or 01
                        Z_prop[self.K,node]=1
                        r = np.random.rand()
                        if r<0.5:
                            Z_prop[self.K-1,node]=1
                            lpsplit =lpsplit + np.log(mk[node]+1-self.sigma) - np.log(self.K+1-self.sigma)
                        else:
                            lpsplit = lpsplit + np.log(self.K-mk[node]) - np.log(self.K+1-self.sigma)
                        lqsplit -=np.log(2)
                    else:
                        r = np.random.rand()
                        if r<(1/3):
                            Z_prop[self.K-1,node]=1
                            #mk is one bigger, and K is one bigger, p0
                            lpsplit = lpsplit + np.log(self.K-mk[node]) - np.log(self.K+1-self.sigma)
                        elif r<(2/3):
                            Z_prop[self.K,node]=1
                            lpsplit = lpsplit + np.log(self.K-mk[node]) - np.log(self.K+1-self.sigma)
                        else:
                            Z_prop[self.K-1,node]=1
                            Z_prop[self.K,node]=1
                            #mk is one bigger, and K is one bigger, p1
                            lpsplit =lpsplit + np.log(mk[node]+1-self.sigma) - np.log(self.K+1-self.sigma)
                        lqsplit -=np.log(3)
                else:
                    #mk is the same and K is one bigger, p0
                    lpsplit = lpsplit + np.log(self.K+1-mk[node]) - np.log(self.K+1-self.sigma)
                        
            #is the resulting proposal valid?
            
            ll_prop = self.loglikZ(Z_prop)
            if not np.isinf(ll_prop):
                ll_old = self.loglikZ()
                #then calculate the acceptance prob
                lqsplit = lqsplit -np.log(np.sum(self.Z[:,sender]))-np.log(np.sum(self.Z[:,receiver]))
                #lqsplit =-np.log(np.sum(self.Z[:,sender]))-np.log(np.sum(self.Z[:,receiver]))
                lqmerge = -np.log(np.sum(self.Z[:,sender])-self.Z[clique_i,sender]+np.sum(Z_prop[:,sender])) - np.log(np.sum(self.Z[:,receiver])-self.Z[clique_i,receiver]+np.sum(Z_prop[:,receiver]))
                
                lpsplit += np.log(self.lamb/(self.K+1))
                laccept = lpsplit -lqsplit+ lqmerge + ll_prop - ll_old
                r = np.log(np.random.rand())
            
                if r<laccept:
                    #pdb.set_trace()
                    #self.checksums
                    self.Z = Z_prop + 0
                    self.K+=1
                #self.checksums()
                
           
            
        else:
            #propose merge
            Z_sum = self.Z[clique_i,:]+self.Z[clique_j,:]
            Z_prop = self.Z+0
            Z_prop[clique_i]= np.minimum(Z_sum,1)
            Z_prop = np.delete(Z_prop,clique_j,0)
            ll_prop = self.loglikZ(Z_prop)
            if not np.isinf(ll_prop):
                #merge OK, proceed
                mk = np.sum(self.Z,0)-Z_sum 
                #calculate the backward probability
                num_affected = np.sum(Z_prop)
                if num_affected<2:
                    pdb.set_trace()
                #lqsplit = -2*np.log(2) - (num_affected-2)*np.log(3)
                #OK now the merge probability
                lqmerge = -np.log(np.sum(self.Z[:,sender]))-np.log(np.sum(self.Z[:,receiver]))
                #lqsplit = lqsplit -np.log(np.sum(self.Z[:,sender])-self.Z[clique_i,sender]-self.Z[clique_j,sender]+1) - np.log(np.sum(self.Z[:,receiver])-self.Z[clique_i,receiver]-self.Z[clique_j,receiver]+1)
                lqsplit =-np.log(np.sum(self.Z[:,sender])-self.Z[clique_i,sender]-self.Z[clique_j,sender]+1) - np.log(np.sum(self.Z[:,receiver])-self.Z[clique_i,receiver]-self.Z[clique_j,receiver]+1)
                #lqsplit +=num_opt*np.log(0.5)
                
                lpsplit =0
                for node in range(self.num_nodes):
                    if Z_sum[node]==0:
                        #mk is the same, and K the same, p0
                        lpsplit = lpsplit + np.log(self.K-mk[node]) - np.log(self.K-self.sigma)
                    elif Z_sum[node]==1:
                        #mk is plus one, and K the same, p0
                        lpsplit = lpsplit + np.log(self.K-mk[node]-1) - np.log(self.K-self.sigma)
                    else:
                        #mk is plus one, and K the same, p2
                        lpsplit = lpsplit + np.log(mk[node]+1-self.sigma) - np.log(self.K-self.sigma)
                        
                lpmerge = np.log(self.K/self.lamb)
                ll_old = self.loglikZ()
                
                laccept = lpmerge -lpsplit +lqsplit - lqmerge + ll_prop - ll_old
                r = np.log(np.random.rand())
                
                if r<laccept:
                    self.Z = Z_prop+0
                    self.K-=1
    def clique_init(self):
        G = nx.Graph()
        G.add_edges_from(self.links)
        all_cliques = list(nx.find_cliques(G))
        self.K = len(all_cliques)
        self.Z = np.zeros((self.K,self.num_nodes))
        for k in range(self.K):
            for n in range(len(all_cliques[k])):
                self.Z[k,all_cliques[k][n]]=1

def sample_from_prior(alpha, sigma, c, num_cliques, pie=None):
    poisson_parameters =np.array([(np.log(alpha) + gammaln(2-sigma)+gammaln(i+1)-gammaln(i+2-sigma)) for i in range(num_cliques)])
    new_counts = np.random.poisson(np.exp(poisson_parameters))
    num_nodes = np.sum(new_counts)
    Z = np.zeros((num_cliques, num_nodes))
    Kplus = 0
    mk = np.zeros(num_nodes)
    for i in range(num_cliques):
        for j in range(Kplus):
            p1 = (mk[j]-sigma)/(i + 1 + c - sigma)
            r = np.random.rand()
            if r<p1:
                Z[i,j]=1
                mk[j]+=1
        Z[i,Kplus:(Kplus+new_counts[i])] =1
        mk[Kplus:(Kplus+new_counts[i])]=1
        Kplus+=new_counts[i]
    
    cic = np.dot(Z.T,Z)
    if pie is None:
        network = np.minimum(cic,1)
    else:
        p0 = (1-pie)**cic
        p1 = 1-p0
        network = np.random.random((num_nodes, num_nodes))
        network = np.triu(network,1)
        network = (network>p0).astype(int)
        network = network + network.T
    np.fill_diagonal(network,0)#network - np.diag(np.diag(network))
    #check for zero rows
    
    empty_nodes = np.where(np.sum(network,1)==0)[0]
    network = np.delete(network,empty_nodes,0)
    network = np.delete(network,empty_nodes,1)
    
    Z = np.delete(Z,empty_nodes,1)
    num_nodes -=len(empty_nodes)
    links = []
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            if network[i,j]==1:
                links.append([i,j])
    
    return Z,network,links

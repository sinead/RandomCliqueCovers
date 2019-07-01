import numpy as np
import networkx as nx
from scipy.special import gammaln
import matplotlib.pyplot as plt
import pdb
import importlib

def poiss_param(alpha, sigma, c, i):
    lpp = np.log(alpha) + gammaln(1+c) + gammaln(i+c+sigma) - gammaln(i+1+c) - gammaln(c+sigma)
    if np.isnan(lpp):
        pdb.set_trace()
    pp = np.exp(lpp)
    return pp

def sample_from_ibp(N, alpha, sigma, c):
    poisson_params = [poiss_param(alpha, sigma, c, n) for n in range(N)]
    Knew = np.array([np.random.poisson(k) for k in poisson_params], dtype=int)
    Z = np.zeros((N, np.sum(Knew)))
    mk = np.zeros(np.sum(Knew), dtype=int)
    Kplus = 0
    for n in range(N):
        for k in range(Kplus):
            p1 = (mk[k] - sigma)/(n + c)
            r = np.random.rand()
            if r < p1:
                Z[n, k] = 1
                mk[k] += 1
        Z[n, Kplus: Kplus + Knew[n]] = 1
        mk[Kplus: Kplus + Knew[n]] = 1
        Kplus += Knew[n]
    return Z

def loglik(mk, Z, alpha, sigma, c):
    N, K = Z.shape
    poiss_params = np.array([poiss_param(alpha, sigma, c, n) for n in range(N)])
    ll = K * np.log(alpha) - np.sum(poiss_params)
    for k in range(K):
        ll = ll + gammaln(mk[k] - sigma) + gammaln(N - mk[k] + c + sigma) + gammaln(1 + c)
        ll = ll - gammaln(1-sigma) - gammaln(c + sigma) - gammaln(N + c)
 
    return ll

# def loglik_batch(mk, Z, alpha, sigma, batch_size):
#     mk_batch = 
#     N, K = Z.shape
#     poiss_params = np.array([poiss_param(alpha, sigma, c, n) for n in range(N)])
#     ll = K * np.log(alpha) - np.sum(poiss_params)
#     for k in range(batch_size):
#         ll = ll + gammaln(mk_batch[k] - sigma) + gammaln(N - mk_batch[k] + c + sigma) + gammaln(1 + c)
#         ll = ll - gammaln(1-sigma) - gammaln(c + sigma) - gammaln(N + c)
 
#     return ll

def loglik_grad(mk, Z, alpha, sigma, c, h):
    N, K = Z.shape
    const = np.sum(np.array([poiss_param(1., sigma, c, n) for n in range(N)]))
    alpha_grad = K/alpha - const
    sigma_grad = 0.5 * (loglik(mk, Z, alpha, sigma+h, c) - loglik(mk, Z, alpha, sigma-h, c)) / h   
    c_grad = 0.5 * (loglik(mk, Z, alpha, sigma, c+h) - loglik(mk, Z, alpha, sigma, c-h)) / h

    return alpha_grad, sigma_grad, c_grad

def backtrack_search(mk, Z, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, curr_step, maxiters=100):
    step = curr_step
    for i in range(maxiters):
        alpha_new = alpha + step * alpha_grad
        sigma_new = sigma + step * sigma_grad
        c_new = c + step * c_grad  
        if alpha_new < 0.0001 or sigma_new < 0.0001 or sigma_new > .9999 or c_new < -sigma_new:
            step = 0.5 * step
        else:
            ll_new = loglik(mk, Z, alpha_new, sigma_new, c_new) 
            if  ll_new >= ll:
                return  step  
            else:
                step = 0.5 * step

    pdb.set_trace()
    print("warning, returning step size {}, step ascent not found".format(step))
    return step
        
def optimize_hypers(Z, alpha, sigma, c, num_iters, print_every = 1, init_stepsize = 0.001, momentum_free_iters=100,
                    momentum=0.1, optimize_alpha = True, optimize_sigma = True, optimize_c = True, h = 0.00001, verbose=True):

    N, K = Z.shape
    mk = np.sum(Z, 0)
    
    ll = loglik(mk, Z, alpha, sigma, c)
    alpha_grad, sigma_grad, c_grad = loglik_grad(mk, Z, alpha, sigma, c, h)
    step = backtrack_search(mk, Z, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, init_stepsize)

    # momentum variables
    d_alpha = 0.
    d_sigma = 0.
    d_c = 0.
    
    if verbose:
        print('iter {:6d}: alpha: {:10.3f}, sigma: {:.3f}, c: {:8.3f}, ll: {:10.3f}, grad: {:10.3f}, {:10.3f}, {:10.3f}, step: {}'. 
          format(0, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, step))
    
    for iter in range(num_iters):
        m = momentum if iter > momentum_free_iters else 0.
        d_alpha = m * d_alpha + step * alpha_grad
        d_sigma = m * d_sigma + step * sigma_grad
        d_c = m * d_c + step * c_grad
        
        alpha = alpha + d_alpha
        sigma = sigma + d_sigma
        c = c + d_c
        
        # gradients again
        ll = loglik(mk, Z, alpha, sigma, c)
        alpha_grad, sigma_grad, c_grad = loglik_grad(mk, Z, alpha, sigma, c, h)        
        step = backtrack_search(mk, Z, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, max(8 * step, 1e-7))        
        
        if iter%print_every == 0:           
            if verbose:
                print('iter {:6d}: alpha: {:10.3f}, sigma: {:.3f}, c: {:8.3f}, ll: {:10.3f}, grad: {:10.3f}, {:10.3f}, {:10.3f}, step: {}'. 
                      format(iter + 1, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, step))
    
    return alpha, sigma, c


def backtrack_search_sigma(mk, Z, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, curr_step, maxiters=100):
    step = curr_step
    for i in range(maxiters):
        sigma_new = sigma + step * sigma_grad
        if sigma_new < 0.0001 or sigma_new > .9999:
            step = 0.5 * step
        else:
            ll_new = loglik(mk, Z, alpha, sigma_new, c) 
            if  ll_new >= ll:
                return  step  
            else:
                step = 0.5 * step
    print("warning, returning step size {}, step ascent not found".format(step))
    return step

def backtrack_search_alpha_c(mk, Z, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, curr_step, maxiters=100):
    step = curr_step
    for i in range(maxiters):
        alpha_new = alpha + step * alpha_grad
        c_new = c + step * c_grad  
        if alpha_new < 0.0001 or c_new < -sigma:
            step = 0.5 * step
        else:
            ll_new = loglik(mk, Z, alpha_new, sigma, c_new) 
            if  ll_new >= ll:
                return  step  
            else:
                step = 0.5 * step
    print("warning, returning step size {}, step ascent nodenst found".format(step))
    return step
        
    

def optimize_hypers2(Z, alpha, sigma, c, num_iters, print_every = 1, init_stepsize = 0.001, momentum_free_iters=100,
                    momentum=0.1, optimize_alpha = True, optimize_sigma = True, optimize_c = True, h = 0.00001, verbose=True):
    "This version does a separete coordinate ascent for sigma and (alpha and c). It should be way more optimal"
    N, K = Z.shape
    mk = np.sum(Z, 0)
    
    ll = loglik(mk, Z, alpha, sigma, c)
    alpha_grad, sigma_grad, c_grad = loglik_grad(mk, Z, alpha, sigma, c, h)
    
    sigma_step = backtrack_search_sigma(mk, Z, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, init_stepsize)
    alpha_c_step = backtrack_search_alpha_c(mk, Z, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, init_stepsize)

    # momentum variables
    d_alpha = 0.
    d_sigma = 0.
    d_c = 0.
    
    if verbose:
        print('iter {:6d}: alpha: {:10.3f}, sigma: {:.3f}, c: {:8.3f}, ll: {:10.3f}, grad: {:10.3f}, {:10.3f}, {:10.3f}, step: {}'. 
          format(0, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, alpha_c_step))
    
    for iter in range(num_iters):
        m = momentum if iter > momentum_free_iters else 0.
        d_alpha = m * d_alpha + alpha_c_step * alpha_grad
        d_sigma = m * d_sigma + sigma_step * sigma_grad
        d_c = m * d_c + alpha_c_step * c_grad
        
        alpha = alpha + d_alpha
        sigma = sigma + d_sigma
        c = c + d_c
        
        # gradients again
        ll = loglik(mk, Z, alpha, sigma, c)
        alpha_grad, sigma_grad, c_grad = loglik_grad(mk, Z, alpha, sigma, c, h)        
        sigma_step = backtrack_search_sigma(mk, Z, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, sigma_step)
        alpha_c_step = backtrack_search_alpha_c(mk, Z, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, alpha_c_step)
    
        if iter%print_every == 0:           
            if verbose:
                print('iter {:6d}: alpha: {:10.3f}, sigma: {:.3f}, c: {:8.3f}, ll: {:10.3f}, grad: {:10.3f}, {:10.3f}, {:10.3f}, step: {}'. 
                      format(iter + 1, alpha, sigma, c, ll, alpha_grad, sigma_grad, c_grad, alpha_c_step))
    
    return alpha, sigma, c
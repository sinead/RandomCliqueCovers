import numpy as np
import networkx as nx
from statsmodels.nonparametric.kde import KDEUnivariate
from collections import Counter
import pandas as pd
import seaborn as sns
from networkx.algorithms import approximation

import sparsedense as spd
import importlib
importlib.reload(spd)



def test_stats(g, mc_size = 500, verbose=True, maxclique=True, return_results=False):
    nodesamp = np.random.choice(list(g.nodes()), mc_size, replace=False)
    
    num_nodes = g.number_of_nodes()
    verbose and print("- num nodes: {:d}".format(num_nodes))
    
    num_edges = g.number_of_edges()
    verbose and print("- num edges: {:d}".format(num_edges))

    edge_node_ratio = num_edges / num_nodes
    verbose and print("- edge node ratio: {:2.2f}".format(edge_node_ratio))
    
    num_triang = sum(nx.triangles(g, i) for i in nodesamp) / 3
    triang_node_ratio = num_triang / mc_size
    verbose and print("- triang node ratio: {:2.2f}".format(triang_node_ratio))
    
    density = 2 * num_edges / num_nodes / (num_nodes - 1)
    verbose and print("- density: {:2.6f}".format(density))
    
    deg = np.mean([nx.degree(g, i) for i in g.nodes()])
    verbose and print("- mean degree: {:2.2f}".format(deg))
        
    #clust_coeff = np.mean([nx.clustering(g, i) for i in nodesamp])
    clust_coeff = approximation.clustering_coefficient.average_clustering(g, trials=mc_size)
    verbose and print("- clustering coefficient: {:2.2f}".format(clust_coeff))

    if maxclique:
        max_clique_node = np.mean([nx.node_clique_number(g, i) for i in nodesamp])
        verbose and print("- mean maximal clique containing node: {:2.2f}".format(max_clique_node))
    else:
        max_clique_node = 0
    
    conn_comp = sorted(list(nx.connected_components(g)), key=lambda x: len(x), reverse=True)
    conn_comp_sizes = [len(xi) for xi in conn_comp]
    verbose and print("- connected component sizes (top 5):", conn_comp_sizes[:5])
    
    short_paths = 0
    for i in range(0): # not used currently...
        pair = np.random.choice(list(conn_comp[0]), 2, replace=False)
        short_paths += nx.dijkstra_path_length(g, *pair) / mc_size
    verbose and print("- mean distance between nodes (largest conn. comp.): {:2.2f}".format(short_paths))
    
    if return_results:
        return num_nodes, num_edges, edge_node_ratio, density, deg, max_clique_node, clust_coeff, conn_comp_sizes[0], short_paths, triang_node_ratio
    else:
        return


def sample_n_from_prior(Ks, alphas, sigmas, cs, n=25):
    prior_samples = []
    for i in range(1, n + 1):
        print("iter", i, " sampling with ( K:", Ks[-i], "  alpha:", alphas[-i], "  sigma:", sigmas[-i], "  c:", cs[-i], ")")
        Z, net, links,  = spd.sample_from_prior(
            alpha = alphas[-i],
            sigma = sigmas[-i],
            c = cs[-i],
            num_cliques = Ks[-i])
        prior_samples.append((Z, net, links))
    return prior_samples

def test_samples_posterior_draws(prior_samples, mc_size=500):
        num_nodes, num_edges, edge_node_ratio, density, deg, max_clique_node, clust_coeff, conn_comp_largest, short_paths = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        n = len(prior_samples)
        for i, (Z, net, links) in enumerate(prior_samples):
            g = nx.Graph()
            g.add_edges_from(links)

            num_nodes_i, num_edges_i, edge_node_ratio_i, density_i, deg_i, max_clique_node_i, clust_coeff_i, conn_comp_i, short_paths_i = \
                test_stats(g, mc_size=mc_size, verbose=False, return_results=True)

            num_nodes += num_nodes_i / n
            num_edges += num_edges_i / n
            edge_node_ratio += edge_node_ratio_i / n
            density += density_i / n
            deg += deg_i / n
            max_clique_node += max_clique_node_i / n
            clust_coeff += clust_coeff_i / n
            conn_comp_largest += conn_comp_i / n / num_nodes_i
            short_paths += short_paths_i / n


        print("- num nodes: {:f}".format(num_nodes))
        print("- num edges: {:f}".format(num_edges))
        print("- edge node ratio: {:2.2f}".format(edge_node_ratio))
        print("- density: {:2.6f}".format(density))
        print("- mean degree: {:2.2f}".format(deg))
        print("- mean maximal clique containing node: {:2.2f}".format(max_clique_node))
        print("- clustering coefficient: {:2.2f}".format(clust_coeff))
        print("- connected component sizes (largest):", conn_comp_largest)
        print("- mean distance between nodes (largest conn. comp.): {:2.2f}".format(short_paths))     


def test_samples_posterior_draws_bnpgraph(dbname, n=25, mc_size=500, verbose=True):
        num_nodes, num_edges, edge_node_ratio, density, deg, max_clique_node, clust_coeff, conn_comp_largest, short_paths = 0., 0., 0., 0., 0., 0., 0., 0., 0.

        for i in range(1, 26):
            links = np.genfromtxt('bnpgraph_runs/' + dbname + '_' + str(i) + '.tsv', delimiter='\t', dtype=int)
            g = nx.Graph()
            g.add_edges_from(links - 1)

            num_nodes_i, num_edges_i, edge_node_ratio_i, density_i, deg_i, max_clique_node_i, clust_coeff_i, conn_comp_i, short_paths_i = \
                test_stats(g, mc_size=mc_size, verbose=False, return_results=True)

            num_nodes += num_nodes_i / n
            num_edges += num_edges_i / n
            edge_node_ratio += edge_node_ratio_i / n
            density += density_i / n
            deg += deg_i / n
            max_clique_node += max_clique_node_i / n
            clust_coeff += clust_coeff_i / n
            conn_comp_largest += conn_comp_i / n / num_nodes_i
            short_paths += short_paths_i / n

        if verbose:
            print("- num nodes: {:f}".format(num_nodes))
            print("- num edges: {:f}".format(num_edges))
            print("- edge node ratio: {:2.2f}".format(edge_node_ratio))
            print("- density: {:2.6f}".format(density))
            print("- mean degree: {:2.2f}".format(deg))
            print("- mean maximal clique containing node: {:2.2f}".format(max_clique_node))
            print("- clustering coefficient: {:2.2f}".format(clust_coeff))
            print("- connected component sizes (largest):", conn_comp_largest)
            print("- mean distance between nodes (largest conn. comp.): {:2.2f}".format(short_paths))      

def test_samples_posterior_draws_krongen(dbname, n=25, mc_size=500, verbose=True):
        num_nodes, num_edges, edge_node_ratio, density, deg, max_clique_node, clust_coeff, conn_comp_largest, short_paths = 0., 0., 0., 0., 0., 0., 0., 0., 0.

        for i in range(1, 26):
            links = np.genfromtxt('krongen_runs/' + dbname + '_{:02d}.tsv'.format(i), delimiter='\t', dtype=int)
            g = nx.Graph()
            for i, j in links:
                if i < j:
                    g.add_edge(i, j)

            num_nodes_i, num_edges_i, edge_node_ratio_i, density_i, deg_i, max_clique_node_i, clust_coeff_i, conn_comp_i, short_paths_i = \
                test_stats(g, mc_size=mc_size, verbose=False, return_results=True)

            num_nodes += num_nodes_i / n
            num_edges += num_edges_i / n
            edge_node_ratio += edge_node_ratio_i / n
            density += density_i / n
            deg += deg_i / n
            max_clique_node += max_clique_node_i / n
            clust_coeff += clust_coeff_i / n
            conn_comp_largest += conn_comp_i / n / num_nodes_i
            short_paths += short_paths_i / n

        if verbose:
            print("- num nodes: {:f}".format(num_nodes))
            print("- num edges: {:f}".format(num_edges))
            print("- edge node ratio: {:2.2f}".format(edge_node_ratio))
            print("- density: {:2.6f}".format(density))
            print("- mean degree: {:2.2f}".format(deg))
            print("- mean maximal clique containing node: {:2.2f}".format(max_clique_node))
            print("- clustering coefficient: {:2.2f}".format(clust_coeff))
            print("- connected component sizes (largest):", conn_comp_largest)
            print("- mean distance between nodes (largest conn. comp.): {:2.2f}".format(short_paths)) 
            
def fit_kde(x, grid):
    resol = len(grid)
    d = np.zeros(resol)
    kde = KDEUnivariate(x)
    kde.fit()
    d = kde.evaluate(grid)    
    return d

def fit_count(x, grid):
    cnt = Counter(x)
    d = np.array([cnt[y] for y in grid]) / len(x)
    return d

def degree_clique_density(prior_samples, grid_deg, grid_clique):
    dens_deg = np.zeros(len(grid_deg))
    dens_clique = np.zeros(len(grid_clique))
    n = len(prior_samples)
    for i, (Z, net, links) in enumerate(prior_samples):
        g = nx.Graph()
        g.add_edges_from(links)

        degs = np.array([nx.degree(g, i) for i in g.nodes()], dtype=float)
        clique = np.array([nx.node_clique_number(g, i) for i in g.nodes()], dtype=float)
        
        dens_deg += fit_kde(degs, grid_deg) / n  
        dens_clique += fit_count(clique, grid_clique) / n  
        
    return dens_deg, dens_clique

def degree_clique_density_orig(g, grid_deg, grid_clique):
    degs = np.array([nx.degree(g, i) for i in g.nodes()], dtype=float)
    clique = np.array([nx.node_clique_number(g, i) for i in g.nodes()], dtype=float)

    dens_deg = fit_kde(degs, grid_deg)
    dens_clique = fit_count(clique, grid_clique) 
        
    return dens_deg, dens_clique


def degree_clique_density_bnpgraph(dbname, grid_deg, grid_clique):
    dens_deg = np.zeros(len(grid_deg))
    dens_clique = np.zeros(len(grid_clique))
    n = 25
    for k in range(n):
        links = np.genfromtxt('bnpgraph_runs/' + dbname + '_' + str(k + 1) + '.tsv', delimiter='\t', dtype=int)
        g = nx.Graph()
        g.add_edges_from(links - 1)


        degs = np.array([nx.degree(g, i) for i in g.nodes()], dtype=float)
        clique = np.array([nx.node_clique_number(g, i) for i in g.nodes()], dtype=float)
        
        dens_deg += fit_kde(degs, grid_deg) / n  
        dens_clique += fit_count(clique, grid_clique) / n  
        
    return dens_deg, dens_clique

def degree_clique_density_krongen(dbname, grid_deg, grid_clique):
    dens_deg = np.zeros(len(grid_deg))
    dens_clique = np.zeros(len(grid_clique))
    n = 25
    for k in range(n):
        links = np.genfromtxt('krongen_runs/' + dbname + '_{:02d}.tsv'.format(k+1), delimiter='\t', dtype=int)
        g = nx.Graph()
        for i, j in links:
            if i < j:
                g.add_edge(i, j)

        degs = np.array([nx.degree(g, i) for i in g.nodes()], dtype=float)
        clique = np.array([nx.node_clique_number(g, i) for i in g.nodes()], dtype=float)
        
        dens_deg += fit_kde(degs, grid_deg) / n  
        dens_clique += fit_count(clique, grid_clique) / n  
        
    return dens_deg, dens_clique
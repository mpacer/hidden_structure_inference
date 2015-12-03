import numpy as np

from .utils import mdp_logsumexp

def sparse_1graph_prior_unnormed(graph,max_edges,sparsity=.5):
    return ((1-sparsity)**(max_edges-len(graph.edges())))*(sparsity)**(len(graph.edges()))

def log_sparse_1graph_prior_unnormed(graph,max_edges,sparsity=.5):
    return ((max_edges-len(graph.edges()))*np.log(1-sparsity))+len(graph.edges())*np.log(sparsity)

def sparse_graphset_prior(graphs,sparsity=.5):
    max_edges = np.max([len(x.edges()) for x in graphs])
    unnormed_p = np.array([sparse_1graph_prior_unnormed(graph,max_edges,sparsity=sparsity) for graph in graphs])
    return unnormed_p/sum(unnormed_p)

def log_sparse_graphset_prior(graphs,sparsity=.5):
    max_edges = np.max([len(x.edges()) for x in graphs])
    unnormed_logp = np.array([log_sparse_1graph_prior_unnormed(graph,max_edges,sparsity=sparsity) for graph in graphs])
    return unnormed_logp - mdp_logsumexp(unnormed_logp)

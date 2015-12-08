import numpy as np
import networkx as nx
import sys
import time
from joblib import Parallel, delayed

from .subgraph_functions import sub_graph_sample
from .sparseprior import log_sparse_graphset_prior 
from .graph_local_classes import InnerGraphSimulation
from .utils import logmeanexp, mdp_logsumexp

class Inference(object):

    def __init__(self):
        self.graphs = None


    def p_graph_given_d(self,graphs,options):
        # sets a catch for all numerical warnings to be treated as errors
        np.seterr(all='raise')
        """
        options includes 
        sparsity: the sparsity argument for the prior 
        param_sample_size: the number of parameters to be sampled
        stigma_sample_size: the number of internal states to be sampled
        data_sets: the different data_sets to be evaluated as likelihoods
        data_p: the probability of each data_set in data_sets
        num_data_samps: number of samples of observed data to be considered in the liklihood
        """
        self.graphs = graphs
        # loglikelihood = np.empty(len(self.graphs))
        param_sample_size= options["param_sample_size"]
        
        loglikelihood = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(self.parameters_monte_carlo_loglik)(graph,
                param_sample_size,options=options) for graph in self.graphs)
        
        # import ipdb; ipdb.set_trace()
        # time_vec = np.empty([len(self.graphs),2])
        # for i,graph in enumerate(self.graphs):
        
        #     loglikelihood[i] = self.parameters_monte_carlo_loglik(graph,param_sample_size,options=options)
        
            # if i in [int(np.floor(j*len(self.graphs))) for j in np.arange(0,1,.1)]:
            #     sys.stdout.write("{:.2%} ".format(i/len(self.graphs)))
            #     sys.stdout.flush()
        # import ipdb; ipdb.set_trace()
        sparsity = options["sparsity"]
        logposterior = self.logposterior_from_loglik_logsparseprior(loglikelihood,sparsity)
        # import ipdb; ipdb.set_trace()
        return graphs,np.exp(logposterior),loglikelihood,options

    def logposterior_from_loglik_logsparseprior(self,loglik,sparsity=.5):
        logp = log_sparse_graphset_prior(self.graphs,sparsity=sparsity)
        unnormed_logposterior = loglik+logp
        return unnormed_logposterior - mdp_logsumexp(unnormed_logposterior)

    def parameters_monte_carlo_loglik(self, graph, param_sample_size, options = None):
        # initialize the dictionary with the scale_free_bounds specified in the options
        init_dict = {"scale_free_bounds":options["scale_free_bounds"]}
        # internal nodes
        gs_in, gp_in = sub_graph_sample(graph, edge_types=["hidden_sample"], param_init=init_dict)
        init_dict["lambda0"]=gp_in.to_dict()["lambda0"]
        gs_out, gp_out = sub_graph_sample(graph, edge_types=['observed'], param_init=init_dict)
        
        param_samples = self._helper_iter_param_sampler(gs_in,gp_in,gs_out,gp_out,param_sample_size,options)
        
        return logmeanexp(np.fromiter(param_samples,dtype=np.float,count=param_sample_size))

    def _helper_iter_param_sampler(self, gs_in,gp_in,gs_out,gp_out,param_sample_size,options):
        # a helper function for sampling parameters in inner graph 
        stigma_sample_size=options["stigma_sample_size"]
        for i in range(param_sample_size):
            # reset the base_rate to None so it is resampled
            update_dict = {"lambda0":None}
            gp_in.update(d=update_dict)
            # sample parameters for inner graph
            gp_in.sample()
            
            # extract base_rate used for inner graph samples to be used for outer graph
            update_dict["lambda0"]=gp_in.to_dict()["lambda0"]
            gp_out.update(d=update_dict)
            # sample parameters for outer graph
            gp_out.sample()
            
            yield self.aux_data_monte_carlo_loglik(gs_in,gp_in,gs_out,gp_out,stigma_sample_size,options=options)

    def gen_simulations(self,gs_in,gp_in,M):
        # builds simulation object and samples it returning an M lengthed list
        inner_simul = InnerGraphSimulation(gs_in,gp_in)
        return inner_simul.sample(M)

    def gen_iter_simulations(self, gs_in,gp_in,M):
        # builds a simulation object and then samples returning an M lengthed generator
        inner_simul = InnerGraphSimulation(gs_in,gp_in)
        return inner_simul.sample_iter(M)


    def gen_iter_simulations_first_only(self, gs_in,gp_in,M):
        # builds a simulation object and then samples returning an M lengthed generator
        inner_simul = InnerGraphSimulation(gs_in,gp_in)
        return inner_simul.sample_iter_solely_first_events(M)


    def aux_data_monte_carlo_loglik(self, gs_in, gp_in, gs_out, gp_out, stigma_sample_size, options=None):
        stigma_sample_size = options["stigma_sample_size"]
        # inner_samp = gen_simulations(gs_in, gp_in, stigma_sample_size)
        # inner_samp = self.gen_iter_simulations(gs_in, gp_in, stigma_sample_size)
        inner_samp = self.gen_iter_simulations_first_only(gs_in, gp_in, stigma_sample_size)
        
        # get arguments to the loglikelihood 
        # data_sets are kinds of data
        data_sets = options["data_sets"]
        
        # data_probs are the probabilities of those data points
        data_probs = options["data_probs"]
        
        # num_data_samps is the number of "sampled" data that we're evaluating it for
        num_data_samps = options["num_data_samps"]

        # get parameters for the relevant nodes to calculate the likelihood 
        obs_dict = gp_out.to_dict()
        
        # build generator for the simulated log_likelihood for a given parameter set
        sim_loglike = (self.cross_entropy_loglik(data_sets, data_probs, num_data_samps, stigma, obs_dict) for stigma in inner_samp)

        return logmeanexp(np.fromiter(sim_loglike,dtype=np.float,count=stigma_sample_size))


    def cross_entropy_loglik(self, data_sets,data_probs, k , aux_data, obs_dict):
        # for a finite set of known kinds of data with known probs
        # we can compute the expected cross-entropy for those kinds of data
        return np.sum([data_probs[i]*k*self.multi_edge_loglik(obs_data, aux_data, obs_dict) for i,obs_data in enumerate(data_sets)])
    

    def multi_edge_loglik(self, obs_data,aux_data,parameters):
        # return np.sum([self.one_edge_loglik(aux_data[i+1],obs_data[i+1],parameters['psi'][i+1],parameters['r'][i+1]) for i in range(len(aux_data)-1)]) 
        
        # special casing for my problem, this needs to be made more general
        # extract non-intervention nodes as we know when the intervention node occurred
        non_int_node_idx = slice(1,4)
        obs_data = obs_data[non_int_node_idx]
        aux_data = aux_data[non_int_node_idx]
        grab = ['psi','r']
        local_dict = {i:parameters[i][1:] for i in parameters if i in grab}
        # # end special casing

        # # loglik of data set is the sum of the loglikelihoods of the individual data points (they're independent)
        
        return np.sum([self.one_edge_loglik(aux_data[i],obs_data[i],local_dict['psi'][i],local_dict['r'][i]) for i in range(len(aux_data))]) 

    def one_edge_loglik(self, cause_time, effect_time, psi, r, T=4.0):
        # if the cause never occurs it occurs at infinity
        if np.isinf(cause_time):
            # it is certain that the effect will not occur if the cause does not occur
            if np.isinf(effect_time):
                return 0
            # it is impossible for the effect to occur if the cause does not occur
            # previous check was elif not np.isinf(effect_time) but that must be true given top if statement
            else:
                return -np.inf

        # if the cause does occur it happens at some time other than infinity
        if not np.isinf(cause_time):
            
            # if the effect doesn't occur, loglik of no events between cause_time and max_time(T)
            if np.isinf(effect_time):
                try:
                    # test to make sure that you won't hit underflow
                    exp_val = np.exp(-r*(T-cause_time))
                except FloatingPointError:
                    # it was a really small number, set it to 0
                    exp_val = 0         
                return -(psi/r)*(1-exp_val)
            
            # if the effect occurred but before the cause_time, that's impossible
            elif effect_time < cause_time: 
                return -np.inf
            # if the effect occurred, after the cause_time, 
            # then loglik of no events from cause_time to effect_time + log of rate function
            else:
                try:
                    # test to make sure that you won't hit underflow
                    exp_val = np.exp(-r*(effect_time-cause_time))
                except FloatingPointError:
                    # it was a really small number, set it to 0
                    exp_val = 0         

                return np.log(psi) - (r*(effect_time-cause_time)) - (psi/r)*(1-exp_val)

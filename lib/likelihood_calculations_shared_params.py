import numpy as np
import networkx as nx
import sys
import time
from joblib import Parallel, delayed
from scipy.misc import logsumexp

from .subgraph_functions import sub_graph_sample, sub_graph_from_edge_type
from .sparseprior import log_sparse_graphset_prior 
from .graph_local_classes import InnerGraphSimulation, GraphParams, GraphStructure
from .utils import logmeanexp, mdp_logsumexp

class Inference(object):

    def __init__(self):
        self.graphs = None

    # def parameter_sampler(self,graphs,options):
    #     # expects top_graph to be the first in the list of graphs
    #     top_graph = graphs[0]
    #     param_sample_size= options["param_sample_size"]
    #     init_dict = {"scale_free_bounds":options["scale_free_bounds"]}
    #     local_g = GraphParams.from_networkx(top_graph)
    #     # i want to get back some kind of object that i can then iterate 
    #     # through on a per graph basis

    def logposterior_from_loglik_logsparseprior(self,loglik,sparsity=.5):
        logp = log_sparse_graphset_prior(self.graphs,sparsity=sparsity)
        unnormed_logposterior = loglik+logp
        try: 
            np.seterr(all='raise')
            unnormed_logposterior - logsumexp(unnormed_logposterior)
            np.seterr(over='raise')
        except RuntimeWarning: 
            import ipdb; ipdb.set_trace()
        
        return unnormed_logposterior - logsumexp(unnormed_logposterior)


    def p_graph_given_d(self,graphs,options):
        # sets a catch for all numerical warnings to be treated as errors
        # np.seterr(all='raise')
        # np.seterr(under='raise')
        np.seterr(over='raise')
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
        self.max_graph = self.graphs[0]
        self.options = options
        num_graphs = len(graphs)
        # loglikelihood = np.empty(len(self.graphs))
        num_params= options["param_sample_size"]

        # generate 1 complete graph with many data structures shared beneath it


        max_graph_params = GraphParams.from_networkx(self.max_graph)
        
        self.param_list = [max_graph_params.sample() for x in range(num_params)]

        # loglikelihood_by_param = np.array(Parallel(n_jobs = -2, 
        #     backend = "multiprocessing", verbose = 10)(
        #     delayed(self._helper_subgraph_loglik)(
        #         max_graph_params.from_dict(params)) for params in self.param_list))
        
        loglikelihood_by_param = np.array(Parallel(n_jobs = -2, 
            backend = "multiprocessing", verbose = 10)(
            delayed(self.subgraph_cross_entropy)(
                max_graph_params.from_dict(params)) for params in self.param_list))


        loglikelihood = logmeanexp(loglikelihood_by_param,axis=0)
        # import ipdb; ipdb.set_trace()
        # time_vec = np.empty([len(self.graphs),2])
        # for i,graph in enumerate(self.graphs):
        
        #     loglikelihood[i] = self.parameters_monte_carlo_loglik(graph,param_sample_size,options=options)
        
            # if i in [int(np.floor(j*len(self.graphs))) for j in np.arange(0,1,.1)]:
            #     sys.stdout.write("{:.2%} ".format(i/len(self.graphs)))
            #     sys.stdout.flush()
        # import ipdb; ipdb.set_trace()
        sparsity = options["sparsity"]
        logposterior = self.logposterior_from_loglik_logsparseprior(loglikelihood,sparsity=sparsity)
        # import ipdb; ipdb.set_trace()
        return graphs,np.exp(logposterior),loglikelihood,self.options,self.param_list

    # def _helper_subgraph_loglik(self,max_graph_params):
    #     return np.array([self.subgraph_cross_entropy(graph,max_graph_params) for graph in self.graphs])


    def subgraph_cross_entropy(self,max_graph_params):
        # sub_graph_params = max_graph_params.subgraph_copy(graph.edges())

        # stigma_sample_size = options["stigma_sample_size"]

        # gs_in = GraphStructure.from_networkx(sub_graph_from_edge_type(graph,
        #     edge_types=["hidden_sample"]))
        # gs_out = GraphStructure.from_networkx(sub_graph_from_edge_type(graph,
        #     edge_types=["observed"]))
        # gp_in = max_graph_params.subgraph_copy(gs_in.edges)
        # gp_out = max_graph_params.subgraph_copy(gs_out.edges)
        n = self.options["num_data_samps"]
        q = self.options["data_probs"]
        δ = self.options["data_sets"]

        # note that q*loglik_from_aux_data should be vector)
        return np.array([n°np.dot(q,self.approx_loglik_from_hidden_states(δ,graph,max_graph_params)) for graph in self.graphs])


    def approx_loglik_from_hidden_states(self,data_sets,graph,max_graph_params):
        K = options["stigma_sample_size"]

        gs_in = GraphStructure.from_networkx(sub_graph_from_edge_type(graph,
            edge_types=["hidden_sample"]))
        gp_in = max_graph_params.subgraph_copy(gs_in.edges)

        hidden_states_iter = self.gen_iter_simulations_first_only(gs_in,gp_in,K)

        return np.array([logmeanexp([self.loglik_with_hidden_states(
                    data_set, hidden_state_sample, graph, max_graph_params)
                    for hidden_state_sample in hidden_states_iter]
                for data_set in data_sets])

    def gen_iter_simulations_first_only(self, gs_in,gp_in,M):
        # builds a simulation object and then samples returning an M lengthed generator
        inner_simul = InnerGraphSimulation(gs_in,gp_in)
        return inner_simul.sample_iter_solely_first_events(M)

    def loglik_with_hidden_states(self, data_set, hidden_state_sample, graph, max_graph_param):

        return np.sum([self.one_edge_loglik(hidden_cause,obs_event,)])


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

        # is this an instantaneous intervention?
        if cause_time - effect_time == 0:
            return 0 

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


### begin valid but deprecated block
# if you were to use these you'd use them together, and they would allow you to
# independently sample the different hidden states, but that will increase variance

    def loglik_from_aux_data(self,data_sets,graph,max_graph_params):
        return np.log(np.array([self.approx_likelihood_from_aux(data_set,
            graph,max_graph_params) for data_set in data_sets])) 

    def approx_likelihood_from_aux(self,data_set,graph,max_params):
        K = options["stigma_sample_size"]

        gs_in = GraphStructure.from_networkx(sub_graph_from_edge_type(graph,
            edge_types=["hidden_sample"]))
        gp_in = max_graph_params.subgraph_copy(gs_in.edges)

        hidden_states_iter = self.gen_iter_simulations_first_only(gs_in,gp_in,K)

        return np.mean([likelihood_with_hidden_states(data_set,
            hidden_state_sample,graph,max_params) for hidden_state_sample 
            in hidden_states_iter])

### end valid but deprecated block



        # transform this to instead return a single value for the loglikelihood of the data, post-perplexity

        # return self.aux_data_monte_carlo_loglik(gs_in,gp_in,gs_out,gp_out,
        #     stigma_sample_size,options=options)


    def gen_simulations(self,gs_in,gp_in,M):
        # builds simulation object and samples it returning an M lengthed list
        inner_simul = InnerGraphSimulation(gs_in,gp_in)
        return inner_simul.sample(M)

    def gen_iter_simulations(self, gs_in,gp_in,M):
        # builds a simulation object and then samples returning an M lengthed generator
        inner_simul = InnerGraphSimulation(gs_in,gp_in)
        return inner_simul.sample_iter(M)



    # def aux_data_monte_carlo_loglik(self, gs_in, gp_in, gs_out, gp_out, stigma_sample_size):
    #     stigma_sample_size = self.options["stigma_sample_size"]
    #     # inner_samp = gen_simulations(gs_in, gp_in, stigma_sample_size)
    #     # inner_samp = self.gen_iter_simulations(gs_in, gp_in, stigma_sample_size)
    #     inner_samp = self.gen_iter_simulations_first_only(gs_in, gp_in, stigma_sample_size)
        
    #     # get arguments to the loglikelihood 
    #     # data_sets are kinds of data
    #     data_sets = self.options["data_sets"]
        
    #     # data_probs are the probabilities of those data points
    #     data_probs = self.options["data_probs"]
        
    #     # num_data_samps is the number of "sampled" data that we're evaluating it for
    #     num_data_samps = self.options["num_data_samps"]

    #     # get parameters for the relevant nodes to calculate the likelihood 
    #     obs_dict = gp_out.to_dict()
        
    #     # build generator for the simulated log_likelihood for a given parameter set
    #     sim_loglike = (self.cross_entropy_loglik(data_sets, data_probs, num_data_samps, stigma, obs_dict) for stigma in inner_samp)

    #     return logmeanexp(np.fromiter(sim_loglike,dtype=np.float,count=stigma_sample_size))


    # def cross_entropy_loglik(self, data_sets,data_probs, k , aux_data, obs_dict):
    #     # for a finite set of known kinds of data with known probs
    #     # we can compute the expected cross-entropy for those kinds of data
    #     return np.sum([data_probs[i]*k*self.multi_edge_loglik(obs_data, aux_data, obs_dict) for i,obs_data in enumerate(data_sets)])
    

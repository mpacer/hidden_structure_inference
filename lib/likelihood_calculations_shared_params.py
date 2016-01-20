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
            # np.seterr(all='raise')
            unnormed_logposterior - logsumexp(unnormed_logposterior)
            # np.seterr(over='raise')
        except RuntimeWarning: 
            import ipdb; ipdb.set_trace()
        
        return unnormed_logposterior - logsumexp(unnormed_logposterior)

    @profile
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

        # Note: some day this will need to change if graphs is made to be an iterator and not a list
        self.max_graph = self.graphs[0]
        self.options = options
        num_graphs = len(graphs)

        # loglikelihood = np.empty(len(self.graphs))
        num_params= options["param_sample_size"]

        # generate 1 complete graph with many data structures shared beneath it
        self.gs_out = GraphStructure.from_networkx(sub_graph_from_edge_type(self.max_graph,
            edge_types=["observed"]))
        
        self.gs_in = [GraphStructure.from_networkx(sub_graph_from_edge_type(graph,
            edge_types=["hidden_sample"])) for graph in graphs]

        max_graph_params = GraphParams.from_networkx(self.max_graph)
        
        self.param_list = [max_graph_params.sample() for x in range(num_params)]

        if self.options["parallel"]:            
            loglikelihood_by_param = np.array(Parallel(n_jobs = -1, 
                backend = "multiprocessing", verbose = 20)(
                delayed(self.subgraph_cross_entropy)(
                    max_graph_params.from_dict(params)) for params in self.param_list))
        else:
            loglikelihood_by_param = np.array([self.subgraph_cross_entropy(max_graph_params.from_dict(params)) 
                for params in self.param_list])
        
        loglikelihood = logmeanexp(loglikelihood_by_param,axis=0)
        
        sparsity = options["sparsity"]
        logposterior = self.logposterior_from_loglik_logsparseprior(loglikelihood,sparsity=sparsity)

        return graphs,np.exp(logposterior),loglikelihood,self.options,self.param_list

    @profile
    def subgraph_cross_entropy(self,max_graph_params):
        n = self.options["num_data_samps"]
        q = np.array(self.options["data_probs"])
        δ = np.array(self.options["data_sets"])
        gp_out = max_graph_params.subgraph_copy(self.gs_out.edges)


        # note that q*approx_loglik_from_hidden_states should be vector)
        return np.array([n*np.dot(q,self.approx_loglik_from_hidden_states(δ,graph,max_graph_params,gp_out,g_idx)) for g_idx,graph in enumerate(self.graphs)])

    def gen_iter_simulations_first_only(self,gs_in,gp_in,K):
        # builds a simulation object and then samples returning an M lengthed generator
        inner_simul = InnerGraphSimulation(gs_in, gp_in)
        return inner_simul.sample_iter_solely_first_events(K)

    @profile
    def gen_simulations_first_only(self,gs_in,gp_in,K):
        # builds a simulation object and then samples returning an M lengthed generator
        inner_simul = InnerGraphSimulation(gs_in, gp_in)
        return inner_simul.sample_solely_first_events(K)

    @profile
    def approx_loglik_from_hidden_states(self,data_sets,graph,max_graph_params,gp_out,g_idx):
        K = self.options["stigma_sample_size"]

        # gs_in = GraphStructure.from_networkx(sub_graph_from_edge_type(graph,
        #     edge_types=["hidden_sample"]))
        gp_in = max_graph_params.subgraph_copy(self.gs_in[g_idx].edges)

        # hidden_states_iter = self.gen_iter_simulations_first_only(self.gs_in[g_idx],gp_in,K)
        hidden_states_iter = self.gen_simulations_first_only(self.gs_in[g_idx],gp_in,K)

        temp_array = np.zeros(shape=(K,data_sets.shape[0]))
        # import ipdb; ipdb.set_trace()
        # for idx, hidden_state_sample in enumerate(hidden_states_iter):
        #     temp_array[idx,:] = np.array([self.loglik_with_hidden_states(data_set,hidden_state_sample,gp_out) for data_set in data_sets])
        # import ipdb; ipdb.set_trace()

        # return value shape should give hidden_sample (first idx) by data_set (idx) summed loglik
        temp_array = self.loglik_with_hidden_states_vectorized(data_sets,hidden_states_iter,gp_out)
        # import IPython; IPython.embed()
        # import ipdb; ipdb.set_trace()
        return logmeanexp(temp_array,axis=0)

    @profile
    def loglik_with_hidden_states_vectorized(self,data_sets,hidden_states,gp_out):

        # K = hidden_states.shape[0]
        # temp_array = np.zeros(shape=(K,data_sets.shape[0]))

        # creates new axis so that it broadcasts correctly

        h_states = hidden_states[:,np.newaxis,:]
        d_sets = data_sets[np.newaxis,:,:]
        # import ipdb; ipdb.set_trace()
        # for i,data_set in enumerate(data_sets):
        #     # temp_array[:,i] = np.sum(self.multi_edge_multisample_loglik_vectorized(
        #     #     hidden_states,data_set,gp_out.psi,gp_out.r),axis=1)
        #     temp_array[:,i] = np.sum(self.multi_edge_loglik_vectorized(
        #         hidden_states,data_set,gp_out.psi,gp_out.r),axis=1)
        

        return np.sum(self.multi_edge_loglik_vectorized(h_states,d_sets,gp_out.psi,gp_out.r),axis=-1)

    @profile
    def loglik_with_hidden_states(self, data_set, hidden_state_sample,gp_out):
        
        #params = zip(hidden_state_sample,data_set,gp_out.psi,gp_out.r)
        #loglik = [self.one_edge_loglik(*p) for p in params]
        #return np.sum(loglik)
        

        return np.sum(self.multi_edge_loglik_vectorized(hidden_state_sample,data_set, gp_out.psi,gp_out.r))

        # return np.sum([self.one_edge_loglik(cause_time, effect_time,psi,r) for 
        #     cause_time, effect_time,psi,r in zip(hidden_state_sample,data_set,gp_out.psi,gp_out.r)])

    @profile
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

    @profile
    def multi_edge_loglik_vectorized(self, cause_time, effect_time, psi, r, T=4.0):
        cause_time, effect_time, psi, r = np.broadcast_arrays(cause_time, effect_time, psi, r)
        out = np.zeros(cause_time.shape)
        cause_inf = np.isinf(cause_time)
        cause_ok = ~cause_inf
        effect_inf = np.isinf(effect_time)
        effect_ok = ~effect_inf
        #out[(cause_time - effect_time) == 0] = 0
        #out[cause_inf & effect_inf] = 0
        out[cause_inf & effect_ok] = -np.inf

        idx = cause_ok & effect_inf
        exp_val = np.exp(-r[idx] * (T - cause_time[idx]))
        
        out[idx] = -(psi[idx] / r[idx]) * (1 - exp_val)

        # out[cause_ok & (effect_time[effect_ok] < cause_time[effect_ok])] = -np.inf
        out[cause_ok & effect_ok & (effect_time < cause_time)] = -np.inf

        idx = cause_ok & (effect_time >= cause_time)
        exp_val = np.exp(-r[idx] * (effect_time[idx] - cause_time[idx]))
        out[idx] = np.log(psi[idx]) - (r[idx] * (effect_time[idx] - cause_time[idx])) - (psi[idx] / r[idx]) * (1 - exp_val)
        # import ipdb; ipdb.set_trace()
        return out
    
    @profile
    def multi_edge_multisample_loglik_vectorized(self, cause_times, effect_time, psi, r, T=4.0):
        
        #passing in many samples, and one dataset for all edges, return matrix of likelihoods
        
        # if this happened then the indexing wouldn't work properly
        assert cause_times.shape[0]!=cause_times.shape[1]

        rs = np.tile(r,[cause_times.shape[0],1])
        psis = np.tile(psi,[cause_times.shape[0],1])
        e_times = np.tile(effect_time,[cause_times.shape[0],1])
        out = np.zeros(cause_times.shape)
        cause_inf = np.isinf(cause_times)
        cause_ok = ~cause_inf
        effect_inf = np.isinf(effect_time)
        effect_ok = ~effect_inf
        #out[(cause_times - e_times) == 0] = 0
        #out[cause_inf & effect_inf] = 0
        out[cause_inf & effect_ok] = -np.inf

        idx = cause_ok & effect_inf
        if idx.any():
            exp_val = np.exp(-rs[idx] * (T - cause_times[idx]))
            # import ipdb; ipdb.set_trace()
            out[idx] = -(psis[idx] / rs[idx]) * (1 - exp_val)

        # import ipdb; ipdb.set_trace()
        out[cause_ok & (effect_time[effect_ok] < cause_times[:,effect_ok])] = -np.inf

        idx = cause_ok & (effect_time >= cause_times) & ((e_times - cause_times)!=0)
        if idx.any():
            exp_val = np.exp(-rs[idx] * (e_times[idx] - cause_times[idx]))
            out[idx] = np.log(psis[idx]) - (rs[idx] * (e_times[idx] - cause_times[idx])) - (psis[idx] / rs[idx]) * (1 - exp_val)

        return out


    def gen_simulations(self,gs_in,gp_in,M):
        # builds simulation object and samples it returning an M lengthed list
        inner_simul = InnerGraphSimulation(gs_in,gp_in)
        return inner_simul.sample(M)

    def gen_iter_simulations(self, gs_in,gp_in,M):
        # builds a simulation object and then samples returning an M lengthed generator
        inner_simul = InnerGraphSimulation(gs_in,gp_in)
        return inner_simul.sample_iter(M)


### begin valid but deprecated block
# if you were to use these you'd use them together, and they would allow you to
# independently sample the different hidden states, but that will increase variance

    # def loglik_from_aux_data(self,data_sets,graph,max_graph_params):
    #     return np.log(np.array([self.approx_likelihood_from_aux(data_set,
    #         graph,max_graph_params) for data_set in data_sets])) 

    # def approx_likelihood_from_aux(self,data_set,graph,max_params):
    #     K = options["stigma_sample_size"]

    #     gs_in = GraphStructure.from_networkx(sub_graph_from_edge_type(graph,
    #         edge_types=["hidden_sample"]))
    #     gp_in = max_graph_params.subgraph_copy(gs_in.edges)

    #     hidden_states_iter = self.gen_iter_simulations_first_only(gs_in,gp_in,K)

    #     return np.mean([likelihood_with_hidden_states(data_set,
    #         hidden_state_sample,graph,max_params) for hidden_state_sample 
    #         in hidden_states_iter])

### end valid but deprecated block

        # transform this to instead return a single value for the loglikelihood of the data, post-perplexity

        # return self.aux_data_monte_carlo_loglik(gs_in,gp_in,gs_out,gp_out,
        #     stigma_sample_size,options=options)



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
    

# def multi_edge_loglik(self, obs_data,aux_data,parameters):
#         # return np.sum([self.one_edge_loglik(aux_data[i+1],obs_data[i+1],parameters['psi'][i+1],parameters['r'][i+1]) for i in range(len(aux_data)-1)]) 
        
#         # special casing for my problem, this needs to be made more general
#         # extract non-intervention nodes as we know when the intervention node occurred
#         non_int_node_idx = slice(1,4)
#         obs_data = obs_data[non_int_node_idx]
#         aux_data = aux_data[non_int_node_idx]
#         grab = ['psi','r']
#         local_dict = {i:parameters[i][1:] for i in parameters if i in grab}
#         # # end special casing

#         # # loglik of data set is the sum of the loglikelihoods of the individual data points (they're independent)
#         return np.sum([self.one_edge_loglik(aux_data[i],obs_data[i],local_dict['psi'][i],local_dict['r'][i]) for i in range(len(aux_data))]) 

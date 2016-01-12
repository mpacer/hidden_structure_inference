import numpy as np
import os

from lib.graph_enumerator import generate_graphs
from lib.node_semantics import Node_Name_Rule, Edge_Semantics_Rule
from lib import config, result_config
from lib.likelihood_calculations_shared_params import Inference
from lib.utils import filename_utility
from lib.misc import cond_to_data
from lib.graph_json_io import json_graph_list_dumps
import time

def main():
    t1 = time.time()
    graph_iter = generate_graphs(**config.generator_dictionary)
    graphs = list(graph_iter)
    for graph in graphs:    
        Node_Name_Rule.graph_semantics_apply(graph,config.node_semantics)
        Edge_Semantics_Rule.graph_semantics_apply(graph,config.edge_semantics)
    
    num_conditions = 4    
    
    options = [config.options]*num_conditions
    for i in range(num_conditions):
        options[i]["data_sets"] = cond_to_data(config.conds[i,:])


    result_graphs = [None]*num_conditions
    result_posteriors = [None]*num_conditions
    result_logliks = [None]*num_conditions
    result_dicts = [None]*num_conditions

    inference_obj = Inference()

    for i in range(num_conditions):
        result_graphs[i], result_posteriors[i], result_logliks[i], result_dicts[i] = inference_obj.p_graph_given_d(graphs,options[i])
    
    # no longer valid edges_of_interest code
    # edges_of_interest = result_config.edges_of_interest
    # for idx,g in enumerate(result_graphs):
    #     for edge in edges_of_interest:
    #         if edge in g.edges():
    #             edges_of_interest[edge]+=result_posteriors[idx]

    result_graphs_strings = [json_graph_list_dumps(g_list) for g_list in result_graphs]

    filename_base = "hidden_structure_results"
    filename = filename_utility(filename_base)
    filename = os.path.join("results",filename)

    with open(filename,'wb') as f:
        np.savez(f,
            g_list_strings=result_graphs_strings, 
            posterior=result_posteriors, 
            loglik=result_logliks,
            init_dict=result_dicts)
        
    elapsed= time.time() - t1
    print(elapsed)

if __name__ == "__main__":
    main()

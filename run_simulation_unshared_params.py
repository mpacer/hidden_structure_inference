import numpy as np
import os

from lib.graph_enumerator import generate_graphs
from lib.node_semantics import Node_Name_Rule, Edge_Semantics_Rule
from lib import config, result_config
from lib.likelihood_calculations import Inference
from lib.utils import filename_utility
import time



def main():
    t1 = time.time()
    graph_iter = generate_graphs(**config.generator_dictionary)
    graphs = list(graph_iter)
    for graph in graphs:    
        Node_Name_Rule.graph_semantics_apply(graph,config.node_semantics)
        Edge_Semantics_Rule.graph_semantics_apply(graph,config.edge_semantics)
        
    inference_obj = Inference()
        
    result_graphs, result_posterior, result_loglik, result_dict = inference_obj.p_graph_given_d(graphs,config.options)
    
    edges_of_interest = result_config.edges_of_interest
    filename_base = "hidden_structure_results"
    filename = filename_utility(filename_base)
    filename = os.path.join("results",filename)



    for idx,g in enumerate(result_graphs):
        for edge in edges_of_interest:
            if edge in g.edges():
                edges_of_interest[edge]+=result_posterior[idx]
    with open(filename,'wb') as f:
        np.savez(f,posterior=result_posterior,loglik=result_loglik,init_dict=result_dict)
    elapsed= time.time() - t1
    print(elapsed)

if __name__ == "__main__":
    main()

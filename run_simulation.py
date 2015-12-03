import numpy as np

from lib.graph_enumerator import generate_graphs
from lib.node_semantics import Node_Name_Rule, Edge_Semantics_Rule
from lib import config, result_config
from lib.likelihood_calculations import Inference


def main():
    graph_iter = generate_graphs(**config.generator_dictionary)
    graphs = list(graph_iter)
    for graph in graphs:    
        Node_Name_Rule.graph_semantics_apply(graph,config.node_semantics)
        Edge_Semantics_Rule.graph_semantics_apply(graph,config.edge_semantics)
        
    inference_obj = Inference()
        
    result_graphs, result_posterior, result_loglik, result_ = inference_obj.p_graph_given_d(graphs,config.options)
    edges_of_interest = result_config.edges_of_interest

    for idx,g in enumerate(result_graphs):
        for edge in edges_of_interest:
            if edge in g.edges():
                edges_of_interest[edge]+=result_posterior[idx]

if __name__ == "__main__":
    main()

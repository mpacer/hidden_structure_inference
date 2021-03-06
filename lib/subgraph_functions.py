import networkx as nx

from .graph_local_classes import GraphStructure, GraphParams

def subgraph_from_edges(G,edge_list,ref_back=True):
    """
    Creates a networkx graph that is a subgraph of G
    defined by the list of edges in edge_list.

    Requires G to be a networkx Graph or DiGraph
    edge_list is a list of edges in either (u,v) or (u,v,d) form
    where u and v are nodes comprising an edge, 
    and d would be a dictionary of edge attributes

    ref_back determines whether the created subgraph refers to back
    to the original graph and therefore changes to the subgraph's 
    attributes also affect the original graph, or if it is to create a
    new copy of the original graph. 
    """
    
    sub_nodes = list({y for x in edge_list for y in x[0:2]})
    edge_list_no_data = [edge[0:2] for edge in edge_list]
    assert all([e in G.edges() for e in edge_list_no_data])

    if ref_back:
        G_sub = G.subgraph(sub_nodes)
        for edge in G_sub.edges():
            if edge not in edge_list_no_data:
                G_sub.remove_edge(*edge)
    else:
        G_sub = G.subgraph(sub_nodes).copy()
        for edge in G_sub.edges():
            if edge not in edge_list_no_data:
                G_sub.remove_edge(*edge)
                
    return G_sub

def sub_graph_from_edge_type(graph,edge_types=None,ref_back=False):
    if edge_types is None:
        edge_types = []
    
    sub_edges = [x for x in graph.edges(data=True) if x[2]['edge_type'] in edge_types]
    return subgraph_from_edges(graph,sub_edges,ref_back=ref_back)


def sub_graph_sample(graph,edge_types=None,param_init=None):
    if param_init is None:
        param_init = {}
    if edge_types is None:
        edge_types = []
        
    sub_graph_struct = GraphStructure.from_networkx(sub_graph_from_edge_type(graph,edge_types=edge_types))
    sub_graph_params = GraphParams.from_structure(sub_graph_struct,init_dict=param_init)
    sub_graph_params.sample()
    
    return (sub_graph_struct,sub_graph_params)

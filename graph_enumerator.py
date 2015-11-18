import networkx as nx
# import json
import filters
import conditions
from networkx.readwrite import json_graph
from utils import powerset
from itertools import combinations

# def clean_json_adj_load(file_name):
#     with open(file_name) as d:
#         json_data = json.load(d)
#     H = json_graph.adjacency_graph(json_data)
#     for edge_here in H.edges():
#         del(H[edge_here[0]][edge_here[1]]["id"])
#     return H

# def clean_json_adj_loads(json_str):
#     json_data = json.loads(json_str)
#     H = json_graph.adjacency_graph(json_data)
#     for edge_here in H.edges():
#         del(H[edge_here[0]][edge_here[1]]["id"])
#     return H


def node_name_edge_picker_2_args(source_node_end,target_node_end,graph):
    f = lambda x: x[0].endswith(source_node_end) and x[1].endswith(target_node_end)
    return [x for x in graph.edges() if f(x)]

def node_name_edge_picker_source_node(source_node_end,graph):
    f = lambda x: x[0].endswith(source_node_end)
    return [x for x in graph.edges() if f(x)]            

def node_name_edge_picker_target_node(target_node_end,graph):
    f = lambda x: x[1].endswith(target_node_end)
    return [x for x in graph.edges() if f(x)]          
            
def completeDiGraph(nodes):
    """
    returns a directed graph with all possible edges
    
    Variables:
    nodes are a list of strings that specify the node names
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    edgelist = list(combinations(nodes,2))
    edgelist.extend([(y,x) for x,y in list(combinations(nodes,2))])
    edgelist.extend([(x,x) for x in nodes])
    G.add_edges_from(edgelist)
    return G

def filter_Graph(G,filter_set):
    graph = G.copy()
    for f in filter_set:
        graph = f(graph)
    return graph

def partialConditionalSubgraphs(G,edge_set,condition_list):
    try: 
        condition_list[0]
    except TypeError:
        raise TypeError("""
        Subsampling from a graph requires passing in a list of conditions encoded
        as first-class functions that accept networkX graphs as an input and return boolean values.""")
    edge_powerset = powerset(edge_set)
 
    for edges in powerset(edge_set):
        G_test = G.copy()
        G_test.remove_edges_from(edges)
        if all([c(G_test) for c in condition_list]):
            yield G_test

def conditionalSubgraphs(G,condition_list):
    try: 
        condition_list[0]
    except TypeError:
        raise TypeError("""
        Subsampling from a graph requires passing in a list of conditions encoded
        as first-class functions that accept networkX graphs as an input and return boolean values.""")
    edge_powerset = powerset(G.edges())
 
    for edges in powerset(G.edges()):
        G_test = G.copy()
        G_test.remove_edges_from(edges)
        if all([c(G_test) for c in condition_list]):
            
            yield G_test



def new_conditional_graph_set(graph_set,condition_list):
    """
    This returns a copy of the old graph_set and a new graph generator which has 
    the conditions in condition_list applied to it.
    
    Warning: This function will devour the iterator that you include as the graph_set input, 
    you need to redeclare the variable as one of the return values of the function.
    
    Thus a correct use would be:    
    a,b = new_conditional_graph_set(a,c)
    
    The following would not be a correct use:
    x,y = new_conditional_graph_set(a,c)
    
    Variables: 
    graph_set is a graph-set generator
    condition_list is a list of first order functions returning boolean values when passed a graph.
    """
    
    try: 
        condition_list[0]
    except TypeError:
        raise TypeError("""
        Subsampling from a graph requires passing in a list of conditions encoded
        as first-class functions that accept networkX graphs as an input and return boolean values.""")
    graph_set_newer, graph_set_test = tee(graph_set,2)
    def gen():
        for G in graph_set_test:
            G_test = G.copy()
            if all([c(G_test) for c in condition_list]):
                yield G_test
    return graph_set_newer, gen()


def set_graph_edge_types(graph,edge_list,edge_type):
    for edge in graph.edges():
        if edge in edge_list:
            nx.set_edge_attributes(graph,"edge_type",{edge:edge_type})
    


def generate_graphs(gen_dict):
    """
    This needs to be edited and improved before this can be released to people outside of myself and collaborators. 
    Currently it is too specific to my problem rather than being a general interface to generate these graphs.
    
    "use if key in dictionary"
    """
    
    nodes = gen_dict["nodes"]
    # explicit_parent_offspring = gen_dict["explicit_parent_offspring"]
    query_edge_set = gen_dict["query_edge_set"]
    reachable_node_pairs = gen_dict["reachable_node_pairs"]

    G = completeDiGraph(nodes)
    
    # Build filters from dictionary
    filter_set = []

    #if it bans self-loops then remove them
    if ("no_self_loops" in gen_dict and 
        gen_dict["no_self_loops"]):
        filter_set.append(extract_remove_self_loops())
        
    # if you are explicitly setting children of certain nodes
    if "explicit_child_parentage" in gen_dict:
        filter_set.append(extract_remove_inward_edges(
            gen_dict["explicit_child_parentage"]))

    # if you are explicitly setting parents of certain nodes
    # this includes 
    if "explicit_parent_offspring" in gen_dict:
        filter_set.append(extract_remove_outward_edges(
            gen_dict["explicit_parent_offspring"]))

    # apply filter set to graph
    G_sub = filter_Graph(G,filter_set)

    """are there any conditions that need to be evalutated on a 
    graph by graph basis?"""
    condition_set = []

    if "reachable_node_pairs" in gen_dict:
        condition_set.append(create_path_complete_condition(
            gen_dict["reachable_node_pairs"]))

    # are there a reduced set of nodes that we'll be querying?
    if "query_edge_set" in gen_dict:
        query_edge_set = gen_dict["query_edge_set"]
    else:
        query_edge_set = nodes

    graph_set = partialConditionalSubgraphs(G_sub,query_edge_set,condition_set)
    working_graphs = list(graph_set)

    return working_graphs


# def add_edge_attribute(graph,edge,attribute_name,attribute_value):
#     graph[edge[0]][edge[1]][attribute_name]=attribute_value
#     pass
    
# def add_multiple_edge_attributes(graph,edge_list,attribute_name,attribute_value):
#     for edge in edge_list:
#         add_edge_attribute(graph,edge,attribute_name,attribute_value)
#     pass

# def add_gamma_attribute_values(graph,edge_list,base_rate,scale):
#     pass

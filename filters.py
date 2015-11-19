import networkx as nx
from registry import Registry

class Filters(Registry):
    """
    Filters are function factories that take arguments 
    (nodes, edges, types, &c.) that specify the particular 
    edges that can be eliminated from consideration in the highest
    level graph that will be enumerated over. 

    These functions will return a function that enacts that graph 
    reduction operation when applied using the function 
    "filter_Graph". That is, it returns a function that will take a
    graph as a
    """
    pass
    

@Filters.register
def explicit_parent_offspring(exceptions_from_removal):
    """
    This covers both no_output and explicit_parent_offspring.
    """
    
    def remove_outward_edges(G):
        graph = G.copy()
        list_of_parents = [x[0] for x in exceptions_from_removal if len(x[1]) > 0]
        list_of_barrens = [x[0] for x in exceptions_from_removal if len(x[1]) == 0]

        for barren in list_of_barrens:
            graph.remove_edges_from([edge for edge in graph.edges() if edge[0] == barren])
            
        for parent in list_of_parents:
            current_edges = graph.out_edges(parent)
            valid_edges = [(x[0],y) for x in exceptions_from_removal if x[0] == parent for y in x[1]]
            graph.remove_edges_from([edge for edge in current_edges if edge not in valid_edges])
            
        return graph
    return remove_outward_edges

@Filters.register
def explicit_child_parentage(exceptions_from_removal):
    """
    This covers both no input and explicit_child_parentage.
    """
    
    def remove_inward_edges(G):
        graph = G.copy()
        list_of_children = [x[0] for x in exceptions_from_removal if len(x[1]) > 0]
        list_of_orphans = [x[0] for x in exceptions_from_removal if len(x[1]) == 0]
        
        for orphan in list_of_orphans:
            graph.remove_edges_from([edge for edge in graph.edges() if edge[1] == orphan])
        
        for child in list_of_children:
            current_edges = graph.in_edges(child)
            valid_edges = [(y,x[0]) for x in exceptions_from_removal if x[0] == child  for y in x[1]]
            graph.remove_edges_from([edge for edge in current_edges if edge not in valid_edges])
        
        return graph
    return remove_inward_edges


@Filters.register
def extract_remove_self_loops():
    def remove_self_loops(G):
        graph = G.copy()
        graph.remove_edges_from(graph.selfloop_edges())
        return graph
    return remove_self_loops



from registry import Registry

class Conditions(Registry):
    """
    Conditions are function factories that take arguments 
    (nodes, edges, types, &c.) that specify the conditions that will 
    need to be met by graphs that cannot be reduced to statements 
    about whether edges or nodes are present (e.g., whether one node 
    A reaches another node B, path_complete(A,B)).

    These functions will return a function that enacts that graph 
    reduction operation when applied using the function 
    "filter_Graph". That is, it returns a function that will take a
    graph as a
    """
    pass

@Conditions.register
def create_path_complete_condition(transmit_node_pairs):
    def path_complete_condition(G):
        return all([nx.has_path(G,x,y) for x,y in transmit_node_pairs])
    return path_complete_condition

@Conditions.register
def create_no_input_node_condition(node_list):
    def no_input_node_condition(G):
        return all([G.in_degree(y)==0 for y in node_list])
    return no_input_node_condition

@Conditions.register
def create_no_self_loops():
    def no_self_loops(G):
        return not(any([(y,y) in G.edges() for y in G.nodes()]))
    return no_self_loops
    
@Conditions.register
def create_explicit_parent_condition(parentage_tuple_list):
    """
    This states for a child node, what its explicit parents are.
    """
    def explicit_parent_condition(G):
        return all(
            [sorted(G.in_edges(y[0])) == sorted([(x,y[0]) for x in y[1]]) 
             for y in parentage_tuple_list])
    return explicit_parent_condition

@Conditions.register
def create_explicit_child_condition(parentage_tuple_list):
    """ 
    This states for a parent node, what its explicit children are.
    """
    def explicit_child_condition(G):
        return all(
            [sorted(G.out_edges(y[0])) == sorted([(y[0],x) for x in y[1]]) 
             for y in parentage_tuple_list])
    return explicit_child_condition

@Conditions.register
def create_no_direct_arrows_condition(node_pair_list):
    def no_direct_arrows_condition(G):
        return not(any([y in G.edges() for y in node_pair_list]))
    return no_direct_arrows_condition

@Conditions.register
def create_no_output_node_condition(node_list):
    def no_output_node_condition(G):
        return all([G.out_degree(y)==0 for y in node_list])
    return no_output_node_condition

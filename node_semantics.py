import nx

def assign_node_semantics(node_name,rule_dict):
    
    for rule_name in rule_dict:
        rule_sem = rule_dict[rule_name]

        if rule_sem["where"] is "prefix":
            infix = rule_sem["infix"]
            code = node_name.split(infix)[0]

        if rule_sem["where"] is "suffix":
            infix = rule_sem["infix"]
            code = node_name.split(infix)[1]

        if code is rule_sem["code"]
            return rule_name

    pass

def enumerate_nodes(graph, rule_dict):
    attrib_dict = {}
    for node in graph.nodes():
        data_type = assign_node_semantics(node,rule_dict)
        attrib_dict[node] = data_type
    nx.set_node_attributes(graph,"data_type",attrib_dict)


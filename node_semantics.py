import networkx as  nx
import re


class Node_Name_Rule(object):

    def __init__(self, node_type, where, infix, code):
        self.node_type = node_type
        self.where = where
        self.infix = infix
        self.code = code

        if where == "suffix":
            self.regex = ".+{}{}".format(infix, code)
        elif where == "prefix":
            self.regex = "{}{}.+".format(code, infix)
        else:
            raise ValueError

    def apply_rule(self, graph):
        for node in graph.nodes():
            if re.match(self.regex, node):
                graph.node[node]["node_type"] = self.node_type

    @classmethod
    def graph_semantics_apply(cls,graph,sem_dict):
        rule_list = [cls(**rule) for rule in sem_dict.values()]
        for r in rule_list:
            r.apply_rule(graph)
        pass


class Edge_Semantics_Rule(object):

    def __init__(self,source_types,target_types,edge_type):
        self.source_types = source_types
        self.target_types = target_types
        self.edge_type = edge_type

    def apply_rule(self, graph):
        for source, target in graph.edges():
            source_ok = not self.source_types or graph.node[source]["node_type"] in self.source_types
            target_ok = not self.target_types or graph.node[target]["node_type"] in self.target_types
            if source_ok and target_ok:
                graph.edge[source][target]["edge_type"] = self.edge_type
            
    @classmethod
    def graph_semantics_apply(cls,graph,sem_dict):
        rule_list = [cls(**rule) for rule in sem_dict.values()]
        for r in rule_list:
            r.apply_rule(graph)
        pass

    # def code_extract(self, node_name):

    #     if self.where is "prefix":
    #         node_val = node_name.split(self.infix)[0]

    #     if self.where is "suffix":
    #         node_val = node_name.split(self.infix)[-1]

    #     if node_val is self.code:
    #         return self.rulename




# def assign_node_semantics(node_name, rule_dict):

#     for rule_name in rule_dict:
#         rule_sem = rule_dict[rule_name]

#         if rule_sem["where"] is "prefix":
#             infix = rule_sem["infix"]
#             code = node_name.split(infix)[0]

#         if rule_sem["where"] is "suffix":
#             infix = rule_sem["infix"]
#             code = node_name.split(infix)[1]

#         if code is rule_sem["code"]:
#             return rule_name

#     pass

# def enumerate_nodes(graph, rule_dict):
#     attrib_dict = {}
#     for node in graph.nodes():
#         data_type = assign_node_semantics(node,rule_dict)
#         #attrib_dict[node] = data_type
#         graph.node[node][some_key] = data_type
#     nx.set_node_attributes(graph,"data_type",attrib_dict)


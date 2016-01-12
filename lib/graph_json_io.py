import json
import networkx as nx
from networkx.readwrite import json_graph
from .utils import open_filename

def json_graph_list_dumps(graph_list):
    return json.dumps([json_graph.adjacency_data(graph) for graph in graph_list])
    

def json_graph_list_dump(graph_list,file):
    with open_filename(file,'w') as f:
        json.dump([json_graph.adjacency_data(graph) for graph in graph_list],f)
    pass

def json_graph_list_loads(json_string):
    js_graph_list = json.loads(json_string)
    return [remove_id_from_json_graphs(json_graph.adjacency_graph(js_graph)) for js_graph in js_graph_list]
    
def json_graph_list_load(file):
    with open_filename(file,'r') as f:
        js_graph_list = json.load(f)
    return [remove_id_from_json_graphs(json_graph.adjacency_graph(js_graph)) for js_graph in js_graph_list]

def remove_id_from_json_graphs(G):
    graph = G.copy()
    for edge in sorted(graph.edges()):
        attr = graph[edge[0]][edge[1]]
        if 'id' in attr:
            del attr['id']
    return graph

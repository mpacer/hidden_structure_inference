import numpy as np
import networkx as nx
from utils import scale_free_sampler

class GraphStructure(object):
    
    def __init__(self, nodes, edges):
        self.nodes = sorted(nodes)
        self.edges = sorted(edges)
       
    @classmethod
    def from_networkx(cls, graph, data=False):
        # cls == GraphStructure
        # graph is a networkx object
        nodes = graph.nodes(data=data)
        edges = graph.edges(data=data)
        obj = cls(nodes, edges)
        return obj
    
    def to_networkx(self):
        graph = nx.DiGraph()
        graph.add_nodes(self.nodes)
        graph.add_edges(self.edges)
        return graph        
        
    def __eq__(self, other):
        if self.nodes != other.nodes:
            return False
        if self.edges != other.edges:
            return False
        return True
    
    def children(self, node):
        return [e for e in self.edges if e[0] == node]

    def parents(self, node):
        return [e for e in self.edges if e[1] == node]


class GraphParams(object):
    
    def __init__(self, n, names = None, p=0.8, 
        scale_free_bounds = (.01,100),
        psi_shape = 1.0, r_shape = 1.0):
        self.n = n             # number of edges
        if names is None:
            self.names = tuple(range(n)) # names of edges
        else:
            self.names = names
        self.p = p             # probability of sending a message
        self.scale_free_lbound = scale_free_bounds[0]
        self.scale_free_ubound = scale_free_bounds[1]
        self.psi_shape = psi_shape
        self.r_shape = r_shape
        self.lambda0 = None    # scale-free parameter
        self.psi = None        # psi edge parameters
        self.r = None          # r edge parameters
        self.mu = None         # psi / r
        
    def init_to_dict(self):
        return {
            "n": self.n,
            "names": self.names,
            "p": self.p,
            "scale_free_bounds": (self.scale_free_lbound, self.scale_free_ubound),
            "psi_shape": self.psi_shape,
            "r_shape": self.r_shape
        }

    def sample(self):
        self.lambda0 = scale_free_sampler(
            lower_bound=self.scale_free_lbound,
            upper_bound=self.scale_free_ubound,
            size=1)
        self.psi = np.random.gamma(shape=1.0, scale=self.lambda0, size=self.n)
        self.r = np.random.gamma(shape=1.0, scale=self.lambda0, size=self.n)
        self.mu = self.psi / self.r
        return self.to_dict()
    
    def to_dict(self):
        return {
            "n": self.n,
            "names": self.names,
            "p": self.p,
            "scale_free_bounds": (self.scale_free_lbound, self.scale_free_ubound),
            "psi_shape": self.psi_shape,
            "r_shape": self.r_shape,
            "lambda0": self.lambda0,
            "psi": self.psi,
            "r": self.r,
            "mu": self.mu
        }
    
    @classmethod
    def from_structure(cls,structure,init_dict=None):
        e_list = sorted(structure.edges)
        if init_dict is None:
            init_dict = {}
        init_dict["names"] = e_list
        g_para = cls(len(e_list), **init_dict)
        return g_para

    @classmethod
    def from_dict(cls, full_dict):
        obj = cls(**full_dict)
        obj.lambda0=d['lambda0']
        obj.psi = d['psi']
        obj.r = d['r']
        return obj
    
    def update(self, d):
        for param, val in d.items():
            if param == 'mu':
                continue
            if hasattr(self, param):
                setattr(self, param, val)
            else:
                raise AttributeError("no such attribute '{}'".format(param))

class InnerGraphSimulation(object):
    """
    right now this only allows a single intervention at the beginning
    what would be better is having an arbitrary set of interventions 
    that can be added at any point during the process
    """

    def __init__(self, structure, params, init_node = None, init_time = 0.0):
        self.structure = structure
        self.params = params
        if init_node is None:
            self.init_node = self.structure.nodes[0]
        self.init_time = init_time
        self._all_events = None
        self._first_events = None
        
    def sample_edge(self, edge, time):
        index = self.structure.edges.index(edge)
        
        # does it occur?
        occurs = np.random.rand() < self.params.p
        if not occurs:
            return []
        
        # how many events?
        num_events = np.random.poisson(lam=self.params.mu[index])
        if num_events == 0:
            return []
        
        # when do those events occur?
        event_times = time + np.random.exponential(scale=self.params.r[index], size=num_events)
        event_times.sort()
        return [(t, edge[1]) for t in event_times]
        
    def _sample(self, first_only=True, max_time=4.0):
        pending = [(self.init_time, self.init_node)]
        self._all_events = []
        self._first_events = None
        if first_only:
            processed_nodes = []

        while len(pending) > 0:
            time, node = pending.pop(0)
            if time >= max_time:
                break

            self._all_events.append((time, node))
            if first_only:
                if node in processed_nodes:
                    continue
                processed_nodes.append(node)

            children = self.structure.children(node)
            for edge in children:
                child_events = self.sample_edge(edge, time)
                if len(child_events) == 0:
                    continue
                pending.extend(child_events)
            pending.sort()
            
        self._all_events.sort()
        self._compute_first_events()
        return self._first_events
    
    def sample(self, k=1, first_only=True, max_time=4.0):
        first_events = np.empty((k, len(self.structure.nodes)))
        for i in range(k):
            first_events[i] = self._sample(first_only=first_only, max_time=max_time)
        return first_events
    
    def _compute_first_events(self):
        first_events = {node: np.inf for node in self.structure.nodes}
        for time, node in self._all_events:
            if first_events[node] < np.inf:
                continue
            first_events[node] = time
        self._first_events = np.array([first_events[node] for node in self.structure.nodes])

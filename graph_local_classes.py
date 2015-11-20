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
    
    def __init__(self, n,  p=0.8):
        self.n = n             # number of edges
        self.p = p             # probability of sending a message
        self.lambda0 = None    # scale-free parameter
        self.psi = None        # psi edge parameters
        self.r = None          # r edge parameters
        self.mu = None         # psi / r
         
    def sample(self):
        self.lambda0 = scale_free_sampler(lower_bound=1/100,upper_bound=100,size=1)
        self.psi = np.random.gamma(shape=1.0, scale=self.lambda0, size=self.n)
        self.r = np.random.gamma(shape=1.0, scale=self.lambda0, size=self.n)
        self.mu = self.psi / self.r
        return self.to_dict()
    
    def to_dict(self):
        return {
            "n": self.n,
            "p": self.p,
            "lambda0": self.lambda0,
            "psi": self.psi,
            "r": self.r,
            "mu": self.mu
        }
    
    @classmethod
    def from_dict(cls, d):
        obj = cls(d['n'], p=d['p'])
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
    
    def __init__(self, structure, params):
        self.structure = structure
        self.params = params
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
        pending = [(0, self.structure.nodes[0])]
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

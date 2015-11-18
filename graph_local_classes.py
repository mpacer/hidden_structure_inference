import numpy as np
import networkx as nx

class GraphStructure(object):
    
    def __init__(self, nodes, edges):
        self.nodes = sorted(nodes)
        self.edges = sorted(edges)
       
    @classmethod
    def from_networkx(cls, graph):
        # cls == Graph
        # graph is a networkx object
        nodes = graph.nodes()
        edges = graph.edges()
        obj = cls(nodes, edges)
        return obj
    
    def to_networkx(self):
        graph = nx.Graph()
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
    
    def __init__(self, n, lambda0=1.0, p=0.8):
        self.n = n             # number of edges
        self.lambda0 = lambda0 # scale-free parameter
        self.p = p             # probability of sending a message
        self.psi = None        # psi edge parameters
        self.r = None          # r edge parameters
        self.mu = None         # psi / r
         
    def sample(self):
        self.psi = np.random.gamma(shape=1.0, scale=self.lambda0, size=self.n)
        self.r = np.random.gamma(shape=1.0, scale=self.lambda0, size=self.n)
        self.mu = self.psi / self.r
        return self.to_dict()
    
    def to_dict(self):
        return {
            "n": self.n,
            "psi": self.psi,
            "r": self.r,
            "lambda0": self.lambda0,
            "p": self.p,
            "mu": self.mu
        }
    
    @classmethod
    def from_dict(cls, d):
        obj = cls(d['n'], lambda0=d['lambda0'], p=d['p'])
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
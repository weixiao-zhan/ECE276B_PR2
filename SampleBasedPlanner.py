from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import math

from util import time_it, XYZ
from env import CollisionCheckingEnvironment, plt

class SampleBasedPlanner(ABC):
    '''
    sample based planner abstract class
    '''
    def __init__(self, start: XYZ, goal: XYZ, env:CollisionCheckingEnvironment, radius = None, n = 5000):
        # build start and goal node
        self.start = start
        self.goal = goal
        self.env = env
        self.n = n
        self.radius = self.opt_radius() if radius is None else radius
        self.graph = nx.Graph()
        self.found_first_path = False # ensure only first exit once
    
    def opt_radius(self):
        d = 1/3
        return 2 * (1+d)**d * (self.env.volume_free() / (4*math.pi/3))**d * (math.log(self.n)/self.n)**d
    
    def build_path(self, map_name = None):
        try:
            cost, path = nx.single_source_dijkstra(self.graph, source=self.start, target=self.goal, weight='weight')
        except nx.exception.NetworkXNoPath:
            cost, path = float('inf'), []
        self.path_np = np.array([[p.x, p.y, p.z] for p in path])
        if map_name is not None:
            np.save(f'./maps/{map_name}_{self.__class__.__name__}', self.path_np)
        return cost

    def near(self, node):
        neighbor = []
        for other in self.graph.nodes:
            if 0 < (other-node).norm() < self.radius:
                neighbor.append(other)
        return neighbor

    def nearest(self, node):
        nearest_node = self.start
        nearest_distance = float('inf')
        for other in self.graph.nodes:
            d = (other-node).norm()
            if 0 < d < nearest_distance:
                nearest_node = other
                nearest_distance = d
        return nearest_node
    
    @staticmethod
    def steer(start, goal, radius):
        dir = goal - start
        norm = dir.norm()
        if norm < radius:
            return goal
        dir = dir * (radius/norm)
        return start + dir
    
    @abstractmethod
    def sample(self):
        pass

    def plot_env(self):
        self.fig, self.ax = self.env.draw_map(self.start, self.goal)

    def plot_graph(self):
        for edge in self.graph.edges():
            x = [edge[0].x, edge[1].x]
            y = [edge[0].y, edge[1].y]
            z = [edge[0].z, edge[1].z]
            self.ax.plot(x, y, z, c='blue',linestyle='--', linewidth=0.5)
        self.fig.canvas.flush_events()

    def plot_path(self):
        self.ax.clear()  # Clear the previous plot
        self.fig, self.ax = self.env.draw_map(self.start, self.goal, self.fig, self.ax)  # Clear the previous plot
        if self.path_np.shape[0] > 0:
            self.ax.plot(self.path_np[:,0],self.path_np[:,1],self.path_np[:,2],'r-')
    
    def check_first_path(self):
        if not self.found_first_path and self.goal in self.graph.nodes:
            print(f"found first path, with cost {self.build_path()}")
            self.found_first_path = True
    
class PRM(SampleBasedPlanner):
    def sample(self, m = 50, 
               first_path_exit = False, plot=True):
        self.graph.add_node(self.start)
        self.graph.add_node(self.goal)
        for i in tqdm(range(self.n)):
            if plot and i%m == 0:
                self.plot_graph()
            new_node = self.env.sample_free()
            self.graph.add_node(new_node)
            for near_node in self.near(new_node):
                if not self.env.line_collide(new_node, near_node):
                    self.graph.add_edge(new_node, near_node, weight=(new_node-near_node).norm())

class RRT(SampleBasedPlanner):
    def __init__(self, start: XYZ, goal: XYZ, env: CollisionCheckingEnvironment, radius=None, n=5000):
        super().__init__(start, goal, env, radius, n)
        self.graph = nx.DiGraph()

    def sample(self, m = 50, step = 0.2,
               first_path_exit = False, plot=True):
        self.graph.add_node(self.start)
        for i in tqdm(range(self.n)):
            if plot and i%m == 0:
                self.plot_graph()
            rand_node, step = (self.env.sample_free(), step) if i % m != 0 else (self.goal, float('inf'))
            nn_node = self.nearest(rand_node)
            new_node = self.steer(nn_node, rand_node, step)
            if not self.env.line_collide(nn_node, new_node):
                self.graph.add_edge(nn_node, new_node, weight=(new_node-nn_node).norm()) # auto add the new_node
                self.check_first_path()
                if first_path_exit and self.found_first_path: return 

class RRTstar(SampleBasedPlanner):
    def __init__(self, start: XYZ, goal: XYZ, env: CollisionCheckingEnvironment, radius=None, n=5000):
        super().__init__(start, goal, env, radius, n)
        self.graph = nx.DiGraph()
        self.cost = dict()
        self.cost[self.start] = 0

    def sample(self, m = 50, step=0.2, 
               first_path_exit = False, plot=True):
        self.graph.add_node(self.start)
        for i in tqdm(range(self.n)):
            if plot and i % m == 0:
                self.plot_graph()
            rand_node, step = (self.env.sample_free(), step) if i % m != 0 else (self.goal, float('inf'))
            nn_node = self.nearest(rand_node)
            new_node = self.steer(nn_node, rand_node, step)
            
            # get near nodes and their connective to new_node
            # cache these results for extend and rewire step
            near_nodes = self.near(new_node)
            line_connectivity = [not self.env.line_collide(near_node, new_node) for near_node in near_nodes]
            line_cost = [(near_node - new_node).norm() for near_node in near_nodes]

            # extend on min cost edge
            min_cost = float("inf")
            min_cost_node = None
            for near_node, connected, cost in zip(near_nodes, line_connectivity, line_cost):
                if connected and self.cost[near_node] + cost < min_cost:
                    min_cost = self.cost[near_node] + cost
                    min_cost_node = near_node
            if min_cost_node is not None:
                self.graph.add_edge(min_cost_node, new_node, weight=(new_node-min_cost_node).norm())
                self.check_first_path()
                if first_path_exit and self.found_first_path: return 
                self.cost[new_node] = min_cost
                # rewire the tree
                for near_node, connected, cost in zip(near_nodes, line_connectivity, line_cost):
                    if connected and self.cost[new_node] + cost < self.cost[near_node]:
                        # found a better route trough new node
                        # remove edge to parent
                        parents = list(self.graph.predecessors(near_node))
                        for parent in parents:
                            self.graph.remove_edge(parent, near_node)
                        # connect to new node
                        self.graph.add_edge(new_node, near_node, weight=(near_node-new_node).norm())
                        # cascade cost reduction though tree
                        cost_decrease = self.cost[near_node] - (self.cost[new_node] + cost)
                        self.cost[near_node] -= cost_decrease
                        for descendant in nx.descendants(self.graph, near_node):
                            self.cost[descendant] -= cost_decrease
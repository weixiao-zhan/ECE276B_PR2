from abc import ABC, abstractmethod
import math
from collections import defaultdict
from functools import cache
from heapdict import heapdict # priority queue for OPEN list
import numpy as np

from util import time_it, XYZ
from env import CollisionCheckingEnvironment, plt

class SearchBasedPlanner(ABC):
    '''
    3D search based planner abstract class
    '''
    def __init__(self, start: XYZ, goal: XYZ, env:CollisionCheckingEnvironment):
        # build start and goal node
        self.start = start
        self.goal = goal
        self.env = env
        # hold cost from start to node
        self.g = defaultdict(lambda: float('inf'))
        # hold shortest path
        self.parent = defaultdict(lambda: None)
        # init open and close set
        self.open_list = heapdict()
        self.close_set = set()
    
    def build_path(self, save_path = None):
        path = []
        current_node = self.goal
        while current_node is not None:
            path.append(current_node)
            current_node = self.parent[current_node]
        self.path_np = np.array([[p.x, p.y, p.z] for p in path[::-1]])
        if save_path is not None:
            np.save(f'./maps/{save_path}_{self.__class__.__name__}', self.path_np)
        return self.path_np

    @abstractmethod
    def search(self):
        pass

    @cache
    def heuristic(self, coord:XYZ):
        return(coord - self.goal).norm()
    
    def plot_env(self):
        self.fig, self.ax = self.env.draw_map(self.start, self.goal)
        
    def plot_path(self):
        self.ax.clear()  # Clear the previous plot
        self.env.draw_map(self.start, self.goal, self.fig, self.ax)
        self.ax.plot(self.path_np[:,0],self.path_np[:,1],self.path_np[:,2],'r-')

class Astar(SearchBasedPlanner):
    def iter_children(self, node: XYZ):
        '''
        yield (child node, cost to child node)
        using self.env.point_collide to ensure child node is valid
        '''
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    child = XYZ(node.x+dx, node.y+dy, node.z+dz)
                    if not self.env.point_collide(child):
                        yield (child, math.sqrt(dx**2+dy**2+dz**2))

    @time_it
    def search(self):
        self.g[self.start] = 0
        self.open_list[self.start] = self.heuristic(self.start)
        while (self.goal not in self.close_set):
            cur, _ = self.open_list.popitem()
            self.close_set.add(cur)
            for child, cost in self.iter_children(cur):
                if child in self.close_set:
                    continue
                if self.g[child] > self.g[cur] + cost:
                    self.g[child] = self.g[cur] + cost
                    self.parent[child] = cur
                    self.open_list[child] = self.g[child]+self.heuristic(child)


'''
class JumpPoint(Astar):
    def __init__(self, start: XYZ, goal: XYZ, env: CollisionCheckingEnvironment):
        super().__init__(start, goal, env)
        self.direction = defaultdict(lambda: XYZ())

    
    def iter_children(self, node: XYZ):
        dir = self.direction[node]
        dir_norm = dir.norm_infinity()

        if dir_norm == 3:
            # natural neighbors
            neighbors = [
                XYZ(node.x+dir.x, node.y+dir.y, node.z+dir.z),

                XYZ(node.x      , node.y+dir.y, node.z+dir.z),
                XYZ(node.x+dir.x, node.y      , node.z+dir.z),
                XYZ(node.x+dir.x, node.y+dir.y, node.z      ),

                XYZ(node.x+dir.x, node.y      , node.z      ),
                XYZ(node.x      , node.y+dir.y, node.z      ),
                XYZ(node.x      , node.y      , node.z+dir.z)            
            ]
            # forces neighbors
            if self.env.point_collide(XYZ(node.x-dir.x, node.y      , node.z       )):
                neighbors.append(XYZ(node.x-dir.x, node.y+dir.y, node.z+dir.z))
                neighbors.append(XYZ(node.x-dir.x, node.y+dir.y, node.z      ))
                neighbors.append(XYZ(node.x-dir.x, node.y      , node.z+dir.z))
            if self.env.point_collide(XYZ(node.x      , node.y-dir.y, node.z       )):
                neighbors.append(XYZ(node.x+dir.x, node.y-dir.y, node.z+dir.z))
                neighbors.append(XYZ(node.x+dir.x, node.y-dir.y, node.z      ))
                neighbors.append(XYZ(node.x      , node.y-dir.y, node.z+dir.z))
            if self.env.point_collide(XYZ(node.x      , node.y      , node.z-dir.z)):
                neighbors.append(XYZ(node.x+dir.x, node.y+dir.y, node.z-dir.z))
                neighbors.append(XYZ(node.x+dir.x, node.y      , node.z-dir.z))
                neighbors.append(XYZ(node.x      , node.y+dir.y, node.z-dir.z))
            if self.env.point_collide(XYZ(node.x      , node.y-dir.y, node.z-dir.y)):
                neighbors.append(XYZ(node.x+dir.x, node.y-dir.y, node.z-dir.y))
            if self.env.point_collide(XYZ(node.x-dir.x, node.y      , node.z-dir.y)):
                neighbors.append(XYZ(node.x-dir.x, node.y+dir.y, node.z-dir.y))
            if self.env.point_collide(XYZ(node.x-dir.x, node.y-dir.y, node.z      )):
                neighbors.append(XYZ(node.x+dir.x, node.y-dir.y, node.z+dir.y))
            
        elif dir_norm == 2:
            zero_index = 0 if dir.x == 0 else (1 if dir.y == 0 else 2)
            one_indices = [0,1,2]
            one_indices.remove(zero_index)

            # natural neighbors
            nn_offsets = [XYZ(dir.x, dir.y, dir.z) for _ in range(3)]
            nn_offsets[1][one_indices[0]] = 0
            nn_offsets[2][one_indices[1]] = 0
            neighbors = [node + offset for offset in nn_offsets]

            # forces neighbors
            fn_3_offsets = [XYZ() for _ in range(2)]
            fn_3_offsets[0][zero_index] = 1
            fn_3_offsets[1][zero_index] = -1
            for offset in fn_3_offsets:
                check_point = node + offset
                if self.env.point_collide(check_point):
                    for nn_offset in nn_offsets:                
                        neighbors.append(check_point + nn_offset)
            fn_1_offsets = [XYZ() for _ in range(3)]
            partial_dir = XYZ()
            partial_dir[one_indices[0]] = dir[one_indices[0]]
            fn_1_offsets[0][one_indices[1]] = -dir[one_indices[1]]
            fn_1_offsets[1][one_indices[1]] = -dir[one_indices[1]]
            fn_1_offsets[2][one_indices[1]] = -dir[one_indices[1]]
            # too complex that I give up
        elif dir_norm == 1:
            one_index = 0
            pass
        else:
            # default case for start node
            pass
'''

import cProfile
import pstats
discretized_start_and_goal = {
    "single_cube": (XYZ(23, 23, 13), XYZ(70, 70, 55)),
    "maze": (XYZ(0, 0, 10), XYZ(120, 120, 50)),
    "window": (XYZ(2, -49, 2), XYZ(60, 180, 30)),
    "tower": (XYZ(25, 40, 5), XYZ(40, 25, 195)),
    "flappy_bird": (XYZ(5, 25, 55), XYZ(190, 25, 55)),
    "room": (XYZ(10, 50, 15), XYZ(90, 70, 15)),
    "monza": (XYZ(5, 10, 49), XYZ(38, 10, 1))
}
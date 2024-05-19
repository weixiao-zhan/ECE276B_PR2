from util import XYZ
import numpy as np
import trimesh
import random

from functools import cache
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class CollisionCheckingEnvironment:
    def __init__(self, env_path, discretize_scale:int = 1) -> None:
        '''
        discretize_scale = 0: not discretize, using floating numbers
        discretize_scale != 0: discretize to int
        '''
        self.boundary, self.blocks = self.load_map(env_path, scale=discretize_scale)
        self.mesh = self.build_block_mesh()
        self.T = int if discretize_scale != 0 else float
    
    def sample(self):
        return XYZ(
            self.T(random.uniform(self.boundary[0].x, self.boundary[1].x)),
            self.T(random.uniform(self.boundary[0].y, self.boundary[1].y)),
            self.T(random.uniform(self.boundary[0].z, self.boundary[1].z))
        )

    def sample_free(self):
        pt = self.sample()
        while self.within_blocks(pt):
            pt = self.sample()
        return pt
    
    def volume_free(self):
        abs_boundary = self.boundary[1] - self.boundary[0]
        volume = abs_boundary.x * abs_boundary.y * abs_boundary.z
        for b_min, b_max, _ in self.blocks:
            abs_block = b_max - b_min
            volume -= abs_block.x * abs_block.y * abs_block.z
        return volume
    
    @cache
    def point_collide(self, pt: XYZ):
        '''
        return true if pt collide with boundary or blocks
        return false o/w
        '''
        return self.within_blocks(pt) or not self.within_boundary(pt)

    def within_boundary(self, pt: XYZ):
        if self.boundary[0].x <= pt.x <= self.boundary[1].x and \
            self.boundary[0].y <= pt.y <= self.boundary[1].y and \
            self.boundary[0].z <= pt.z <= self.boundary[1].z:
            return True
        return False
    
    def within_blocks(self, pt: XYZ):
        for b_min, b_max, _ in self.blocks:
            if b_min.x <= pt.x <= b_max.x and \
                b_min.y <= pt.y <= b_max.y and \
                b_min.z <= pt.z <= b_max.z:
                return True
        return False


    def line_collide(self, start: XYZ, end:XYZ):
        # Create a line object in trimesh
        ray_origin = np.array([[start.x, start.y, start.z]])
        ray_direction = np.array([[end.x - start.x, end.y-start.y, end.z-start.z]])

        # Perform ray-mesh intersection
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=ray_origin,
            ray_directions=ray_direction
        )

        # Check if there is an intersection and if it's within the segment bounds
        for location in locations:
            # Calculate the parameter t for the intersection point location on the segment
            t = np.dot(location - ray_origin.squeeze(), ray_direction[0]) / np.dot(ray_direction[0], ray_direction[0])
            if 0 <= t <= 1:
                return True
            
        return False
    
    @staticmethod
    def load_map(env_path, scale):
        '''
        Loads the bounady and blocks from map file fname.
        
        boundary = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
        
        blocks = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'],
                ...,
                ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
        '''
        mapdata = np.loadtxt(env_path,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                        'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
        blockIdx = mapdata['type'] == b'block'
        boundary_data = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
        blocks_data = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
        if scale == 0:
            boundary = [
                XYZ(*(boundary_data[0][0:3])), 
                XYZ(*(boundary_data[0][3:6])), 
                boundary_data[0][6:9]
            ]
            blocks = [
                (
                    XYZ(*(block[0:3])), 
                    XYZ(*(block[3:6])), 
                    block[6:9]
                ) for block in blocks_data
            ]
        else:
            boundary = [
                XYZ(*(boundary_data[0][0:3]*scale).astype(int)), 
                XYZ(*(boundary_data[0][3:6]*scale).astype(int)), 
                boundary_data[0][6:9]
            ]
            blocks = [
                (
                    XYZ(*(block[0:3]*scale).astype(int)), 
                    XYZ(*(block[3:6]*scale).astype(int)), 
                    block[6:9]
                ) for block in blocks_data
            ]
        return boundary, blocks

    def build_block_mesh(self):
        # Create a list to store all vertices and faces
        all_vertices = []
        all_faces = []
        
        # Iterate over each AABB
        for xyz_min, xyz_max, _ in self.blocks:
            xmin, ymin, zmin = xyz_min.x, xyz_min.y, xyz_min.z
            xmax, ymax, zmax = xyz_max.x, xyz_max.y, xyz_max.z

            # Define the 8 vertices of the AABB
            vertices = np.array([
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmax, ymax, zmin],
                [xmin, ymax, zmin],
                [xmin, ymin, zmax],
                [xmax, ymin, zmax],
                [xmax, ymax, zmax],
                [xmin, ymax, zmax]
            ])
            
            # Define the 12 triangles (2 per face of the AABB)
            faces = np.array([
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [2, 3, 7], [2, 7, 6],
                [1, 2, 6], [1, 6, 5],
                [0, 3, 7], [0, 7, 4]
            ])
            
            # Adjust face indices for the current set of vertices
            faces += len(all_vertices)
            
            # Append vertices and faces to the lists
            all_vertices.extend(vertices)
            all_faces.extend(faces)
        
        # Create a single mesh with all AABBs
        mesh = trimesh.Trimesh(vertices=np.array(all_vertices), faces=np.array(all_faces))
        return mesh
    
    def draw_map(self, start = None, goal = None,
                 fig = plt.figure(), ax = None):
        '''
        Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
        '''
        ax = fig.add_subplot(111, projection='3d') if ax is None else ax
        
        self.draw_block_list(ax, self.blocks)
        if start is not None:
            ax.plot([start.x], [start.y], [start.z], 'ro', markersize=7, markeredgecolor='k')
        if goal is not None:
            ax.plot([goal.x], [goal.y], [goal.z], 'go', markersize=7, markeredgecolor='k')  
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(self.boundary[0].x, self.boundary[1].x)
        ax.set_ylim(self.boundary[0].y, self.boundary[1].y)
        ax.set_zlim(self.boundary[0].z, self.boundary[1].z)
        return fig, ax
    
    @staticmethod
    def draw_block_list(ax,blocks):
        '''
        Subroutine used by draw_map() to display the environment blocks
        '''
        v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype='float')
        f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
        
        n = len(blocks)
        vl = np.zeros((8 * n, 3))
        fl = np.zeros((6 * n, 4), dtype='int64')
        fcl = np.zeros((6 * n, 3))
        
        for k, (min_xyz, max_xyz, color) in enumerate(blocks):
            d = max_xyz - min_xyz
            vl[k * 8:(k + 1) * 8, :] = v * np.array([d.x, d.y, d.z]) + np.array([min_xyz.x, min_xyz.y, min_xyz.z])
            fl[k * 6:(k + 1) * 6, :] = f + k * 8
            fcl[k * 6:(k + 1) * 6, :] = np.array(color) / 255

        if isinstance(ax, Poly3DCollection):
            ax.set_verts(vl[fl])
        else:
            pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
            pc.set_facecolor(fcl)
            h = ax.add_collection3d(pc)
            return h
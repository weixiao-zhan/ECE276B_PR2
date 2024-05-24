from env import CollisionCheckingEnvironment, np
from util import XYZ
import matplotlib.pyplot as plt
from cycler import cycler
colors = plt.get_cmap('tab20').colors
cycle = cycler(color=colors)
plt.rcParams['axes.prop_cycle'] = cycle

start_and_goal = {
    "single_cube": (XYZ(2.3, 2.3, 1.3), XYZ(7.0, 7.0, 5.5)),
    "maze": (XYZ(0.0, 0.0, 1), XYZ(12.0, 12.0, 5.0)),
    "window": (XYZ(2.0, -4.9, 2.0), XYZ(6.0, 18.0, 3.0)),
    "tower": (XYZ(2.5, 4.0, 0.5), XYZ(4.0, 2.5, 19.5)),
    "flappy_bird": (XYZ(0.5, 2.5, 5.5), XYZ(19.0, 2.5, 5.5)),
    "room": (XYZ(1.0, 5.0, 1.5), XYZ(9.0, 7.0, 1.5)),
    "monza": (XYZ(0.5, 1.0, 4.9), XYZ(3.8, 1.0, 0.1))
}

map_names = ["single_cube", "maze", "window", "tower", "flappy_bird", "room", "monza"]
method_names = ["Astar", "PRM", "RRT", "RRTstar"]

for map_name in map_names:
    env = CollisionCheckingEnvironment(f'./maps/{map_name}.txt', discretize_scale=0)
    fig = plt.figure(figsize=(8, 8), dpi=100)  # Manually create the figure
    ax = fig.add_subplot(111, projection='3d')  # Manually create the axis
    env.draw_map(start=start_and_goal[map_name][0], 
                 goal=start_and_goal[map_name][1],
                 fig = fig, ax = ax)
    for method_name in method_names:
        path_np = np.load(f"./maps/{map_name}_{method_name}.npy")
        if method_name == "Astar":
            path_np = path_np/8
        if path_np.shape[0] > 0:
            ax.plot(path_np[:,0],path_np[:,1],path_np[:,2],label=method_name)
        if method_name == "Astar":
            ax.plot(path_np[:,0],path_np[:,1],path_np[:,2])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    # plt.savefig(map_name)
    plt.show(block=True)
    # break
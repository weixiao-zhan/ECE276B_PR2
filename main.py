from util import XYZ
from env import CollisionCheckingEnvironment, plt, np
from SearchBasedPlanner import Astar
from SampleBasedPlanner import PRM, RRT, RRTstar

import cProfile
import pstats

PROFILE = False
PLOT = False
start_and_goal = {
    "single_cube": (XYZ(2.3, 2.3, 1.3), XYZ(7.0, 7.0, 5.5)),
    "maze": (XYZ(0.0, 0.0, 1), XYZ(12.0, 12.0, 5.0)),
    "window": (XYZ(2.0, -4.9, 2.0), XYZ(6.0, 18.0, 3.0)),
    "tower": (XYZ(2.5, 4.0, 0.5), XYZ(4.0, 2.5, 19.5)),
    "flappy_bird": (XYZ(0.5, 2.5, 5.5), XYZ(19.0, 2.5, 5.5)),
    "room": (XYZ(1.0, 5.0, 1.5), XYZ(9.0, 7.0, 1.5)),
    "monza": (XYZ(0.5, 1.0, 4.9), XYZ(3.8, 1.0, 0.1))
}

def run_search(search_based_planner, map_name, discretize_scale = 5):
    env = CollisionCheckingEnvironment(f'./maps/{map_name}.txt', discretize_scale=discretize_scale)
    start, goal = start_and_goal[map_name]
    start = (start * discretize_scale).astype(int)
    goal = (goal * discretize_scale).astype(int)
    planner = search_based_planner(start, goal, env)

    profiler = cProfile.Profile()
    if PROFILE:
        profiler.enable()
    planner.search()
    if PROFILE:
        profiler.disable()
        # Save profiling results to a file
        profiler.dump_stats('profile_output')
        p = pstats.Stats('profile_output')
        p.sort_stats('tottime').print_stats(10)  # Sort by tottime and print top 10 functions
    
    print("path cost = ", planner.build_path(map_name)/discretize_scale)
    print(f"close set size: {len(planner.close_set)}, open set size: {len(planner.open_list)}")


def run_sample(sample_based_planner, map_name):
    env = CollisionCheckingEnvironment(f'./maps/{map_name}.txt', discretize_scale=0)
    start, goal = start_and_goal[map_name]
    planner = sample_based_planner(start, goal, env, n=2000)
    
    if PLOT:
        planner.plot_env()

    profiler = cProfile.Profile()
    if PROFILE:
        profiler.enable()
    planner.sample(plot = PLOT)
    if PROFILE:
        profiler.disable()
        # Save profiling results to a file
        profiler.dump_stats('profile_output')
        p = pstats.Stats('profile_output')
        p.sort_stats('tottime').print_stats(10)  # Sort by tottime and print top 10 functions

    print("path cost =", planner.build_path(map_name))
    print(f"node size {len(planner.graph.nodes)}, edge size {len(planner.graph.edges)}")

    if PLOT:
        planner.plot_path()
        plt.show(block=True)

if __name__ == "__main__":
   for map_name in start_and_goal:  
        print(f"\nmap: {map_name}")
        print("running Astar, scale = 5")
        run_search(Astar, map_name, 5)
        print("running Astar, scale = 8")
        run_search(Astar, map_name, 8)
        print("running PRM")
        run_sample(PRM, map_name)
        print("running RRT")
        run_sample(RRT, map_name)
        print("running RRTstar")
        run_sample(RRTstar, map_name)
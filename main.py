from util import XYZ
from env import CollisionCheckingEnvironment, plt, np
from SearchBasedPlanner import Astar
from SampleBasedPlanner import PRM, RRT, RRTstar

import cProfile
import pstats

PROFILE = False

start_and_goal = {
    "single_cube": (XYZ(2.3, 2.3, 1.3), XYZ(7.0, 7.0, 5.5)),
    "maze": (XYZ(0.0, 0.0, 1), XYZ(12.0, 12.0, 5.0)),
    "window": (XYZ(2.0, -4.9, 2.0), XYZ(6.0, 18.0, 3.0)),
    "tower": (XYZ(2.5, 4.0, 0.5), XYZ(4.0, 2.5, 19.5)),
    "flappy_bird": (XYZ(0.5, 2.5, 5.5), XYZ(19.0, 2.5, 5.5)),
    "room": (XYZ(1.0, 5.0, 1.5), XYZ(9.0, 7.0, 1.5)),
    "monza": (XYZ(0.5, 1.0, 4.9), XYZ(3.8, 1.0, 0.1))
}
F = "maze"

def run_sample(f):
    env = CollisionCheckingEnvironment(f'./maps/{f}.txt', discretize_scale=0)
    start, goal = start_and_goal[f]
    planner = RRTstar(start, goal, env, n = 4_000)

    planner.plot_env()

    profiler = cProfile.Profile()
    if PROFILE:
        profiler.enable()
    planner.sample(plot=not PROFILE)
    if PROFILE:
        profiler.disable()
        # Save profiling results to a file
        profiler.dump_stats('profile_output')
        p = pstats.Stats('profile_output')
        p.sort_stats('tottime').print_stats(10)  # Sort by tottime and print top 10 functions

    planner.build_path()
    planner.plot_path()
    plt.show(block=True)


def run_search(f, discretize_scale = 10):
    env = CollisionCheckingEnvironment(f'./maps/{f}.txt', discretize_scale=discretize_scale)
    start, goal = start_and_goal[f]
    start = (start * discretize_scale).astype(int)
    goal = (goal * discretize_scale).astype(int)
    planner = Astar(start, goal, env)

    planner.plot_env()

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

    planner.build_path()
    planner.plot_path()
    plt.show(block=True)

if __name__ == "__main__":
    run_search(F)
    # run_sample(F)
# def test_single_cube(verbose = True):
#     print('Running single cube test...\n') 
#     start = XYZ(2.3, 2.3, 1.3)
#     goal = XYZ(7.0, 7.0, 5.5)
#     success, pathlength = runtest('./maps/single_cube.txt', start, goal, verbose)
#     print('Success: %r'%success)
#     print('Path length: %d'%pathlength)
#     print('\n')
  
  
# def test_maze(verbose = True):
#     print('Running maze test...\n') 
#     start = XYZ(0.0, 0.0, 1.0)
#     goal = XYZ(12.0, 12.0, 5.0)
#     success, pathlength = runtest('./maps/maze.txt', start, goal, verbose)
#     print('Success: %r'%success)
#     print('Path length: %d'%pathlength)
#     print('\n')

    
# def test_window(verbose = True):
#     print('Running window test...\n') 
#     start = XYZ(0.2, -4.9, 0.2)
#     goal = XYZ(6.0, 18.0, 3.0)
#     success, pathlength = runtest('./maps/window.txt', start, goal, verbose)
#     print('Success: %r'%success)
#     print('Path length: %d'%pathlength)
#     print('\n')

  
# def test_tower(verbose = True):
#     print('Running tower test...\n') 
#     start = XYZ(2.5, 4.0, 0.5)
#     goal = XYZ(4.0, 2.5, 19.5)
#     success, pathlength = runtest('./maps/tower.txt', start, goal, verbose)
#     print('Success: %r'%success)
#     print('Path length: %d'%pathlength)
#     print('\n')

     
# def test_flappy_bird(verbose = True):
#     print('Running flappy bird test...\n') 
#     start = XYZ(0.5, 2.5, 5.5)
#     goal = XYZ(19.0, 2.5, 5.5)
#     success, pathlength = runtest('./maps/flappy_bird.txt', start, goal, verbose)
#     print('Success: %r'%success)
#     print('Path length: %d'%pathlength) 
#     print('\n')

  
# def test_room(verbose = True):
#     print('Running room test...\n') 
#     start = XYZ(1.0, 5.0, 1.5)
#     goal = XYZ(9.0, 7.0, 1.5)
#     success, pathlength = runtest('./maps/room.txt', start, goal, verbose)
#     print('Success: %r'%success)
#     print('Path length: %d'%pathlength)
#     print('\n')


# def test_monza(verbose = True):
#     print('Running monza test...\n')
#     start = XYZ(0.5, 1.0, 4.9)
#     goal = XYZ(3.8, 1.0, 0.1)
#     success, pathlength = runtest('./maps/monza.txt', start, goal, verbose)
#     print('Success: %r'%success)
#     print('Path length: %d'%pathlength)
#     print('\n')

# def test_all():
#     test_single_cube()
#     test_maze()
#     test_flappy_bird()
#     test_monza()
#     test_window()
#     test_tower()
#     test_room()
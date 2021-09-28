"""
Divide and conquer Algorithm (Approximate) 
The other general strategy for TSP is divide and conquer.
Suppose we have an algorithm, like alltours_tsp, that is inefficient for large n (is O(n!)).
We cann't apply it directly to a large set of cities, but we can divide the problem into smaller
piecies, and then combile those piecies:
1. split the set of cities in half
2. find a tour for each half
3. join those two tours into one
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
import urllib
from map import *
from utils import *

def extent(numbers):
    return max(numbers) - min(numbers)

def rotations(sequence):
    """All possible rotations of a sequence."""
    return [sequence[i:]+sequence[:i] for i in range(len(sequence))]

def split_cities(cities):
    """Split cities vertically if map is wider; horizontally if map is taller."""
    width, height = extent([c.x for c in cities]) ,extent([c.y for c in cities])
    key = 'x' if (width > height) else 'y'
    cities = sorted(cities,key=lambda c: getattr(c,key))
    mid = len(cities) // 2
    return frozenset(cities[:mid]), frozenset(cities[mid:])

def join_tours(tour1, tour2):
    """Consider all ways of joining the two tours together, and pick the shortest."""
    segements1, segments2 = rotations(tour1), rotations(tour2)
    tours = [s1 + s2 for s1 in segements1 for s in segments2 for s2 in (s, s[::-1])]
    return shortest_tour(tours)


def dq_tsp(cities):
    """
    Find a tour by divide and conquer: if number of cities is 3 or less,
    any tour is optimal. Otherwise, split the cities in half, solve each
    half recursively, then join those two tours together.
    """
    if len(cities) <=3:
        return list(cities)
    else:
        Cs1, Cs2 = split_cities(cities)
        return join_tours(dq_tsp(Cs1),dq_tsp(Cs2))


# main loop
if __name__ == '__main__':
    #cities = USA_landmarks_map()
    args = get_args()
    cities = Random_cities_map(args.size)
    if args.map == "USA":
        cities = USA_landmarks_map()
    if args.map == "USA-big":
        cities = USA_big_map()
    #plot_tsp(nn_tsp,cities)
    #plot_tsp(repeated_nn_tsp,cities)
    #plot_tsp(altered_nn_tsp,cities)
    plot_tsp(dq_tsp,cities)

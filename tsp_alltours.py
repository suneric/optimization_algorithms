"""
Brute-Force Solution (Optimal, inefficient) 
Generate all possible tours of the cities, and choose the shortest tour
(this is inefficient for large sets of cities)
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
import urllib
from map import *
from utils import *

#alltours = itertools.permutations
def alltours(cities):
    """Return a list of tours, each of permutation of cities, but each one starting with the same city.
    There are n! tours of n cities. But some of the tours are redudant. So re-asssembling a tour from
    the start city and the rest."""
    start = first(cities)
    all = [[start]+list(rest) for rest in itertools.permutations(cities-{start})]
    print("there are {} cities and {} tours".format(len(cities), len(all)))
    #print(all)
    return all

"""
In genral, this function loos at (n-1)! tours for an n-city problem, and each tour has n cities.
So the total time required for n cities should be roughly proportional to n!
This means that the time grows rapidly with the number of cities. Realy rapidly
"""
def alltours_tsp(cities):
    """
    If a tour is a sequence of cities, then all the tours are permutations of the set of all cities.
    A fuction to generate all permutations of a set is already provide in Python's standard itertools.
    """
    return shortest_tour(alltours(cities))

# main loop
if __name__ == '__main__':
    #cities = USA_landmarks_map()
    args = get_args()
    cities = Random_cities_map(args.size)
    if args.map == "USA":
        cities = USA_landmarks_map()
    plot_tsp(alltours_tsp,cities)

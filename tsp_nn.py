"""
Nearest Neighbor Algorithm (Approximate)
Settle for a tour that is short, but not guaranteed to be shortest.(for saving computation time)
There are more sophisticated approximate algorithms that can handle hundreds of thousands of cities
and come with 0.01% or better of the shortest passible tour.
A general plans to create a tour:
Nearest Neighbor Algorithm: make the tour go from a city to its nearest neighbor. Repeat
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
import urllib
from map import *
from utils import *


def nearest_neighbor(A, cities):
    """Find the city in cities that is nearest to city A."""
    return min(cities, key=lambda c: distance(c,A))

"""
Limitation: We might see a very long edge at the endo of nn_tsp, because there are
no remaining cities near by. In a way, this just seems like bad luck- we started in
a place that left us with no good choices at the end.
"""
def nn_tsp(cities,start=None):
    """
    Start the tour at the first city, at each step extend the tour
    by moving from the previous city to the nearest neighboring city, C,
    that has not yet been visited.
    """
    if start is None: start = first(cities)
    tour = [start]
    unvisited = set(cities - {start})
    while unvisited:
        C = nearest_neighbor(tour[-1],unvisited)
        tour.append(C)
        unvisited.remove(C)
    return tour


"""
An easy way to apply repetition strategy to imporve nearest neighbors:
Repeated Nearest Neighbor Algorithm: For each of the cities, run the nearest algorithm
with that city as the starting point, and chosse the resulting tour with the shortest total
distance.
Given an optimal argument specifying the number of different cities to try starting from.
when dealing with large number of cities (such as 1000, as we don't really want to run 1000
times of nn_tsp)
"""
def repeated_nn_tsp(cities, repetitions=100):
    return shortest_tour(nn_tsp(cities,start) for start in sample(cities, repetitions))


def altered_nn_tsp(cities):
    """Run nearest neighbor TSP algorithm, and alter the results by reversing segements."""
    return alter_tour(nn_tsp(cities))

def repeated_altered_nn_tsp(cities,repetitions=20):
    return shortest_tour(alter_tour(nn_tsp(cities,start)) for start in sample(cities, repetitions))

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
    plot_tsp(repeated_altered_nn_tsp,cities)

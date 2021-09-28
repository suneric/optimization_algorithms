"""
Greedy algorithm (approximate)
Settle for a tour that is short, but not guaranteed to be shortest.(for saving computation time)
There are more sophisticated approximate algorithms that can handle hundreds of thousands of cities
and come with 0.01% or better of the shortest passible tour.
A general plans to create a tour:
Greedy Algotithm: Find the shortest distance between any two cities and include that edge in tour. Repeat
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
import urllib
from map import *
from utils import *

"""
At every step, Greedy Algorithm greedily adds to the tour the edge that is shortest (even if that
is not best in terms of long-range planning). The nearest neighbor algorithm always extended the tour
by adding on to the end. The greedy algorithm is different in that it doesn't have a notion of end of
the tour; instead, it keeps a set of partial segments.
    Maintain a set of segments; initially each city defines its own 1-city segment. Find the
    shortest possible edge that connects two endpoints segments, and join those segment with that edge.
    Repeat until we form a segment that tours all the cities.
"""
def join_endpoints(endpoints,A,B):
    Asegment, Bsegment = endpoints[A], endpoints[B]
    if Asegment[-1] is not A: Asegment.reverse()
    if Bsegment[0] is not B: Bsegment.reverse()
    Asegment.extend(Bsegment)
    del endpoints[A], endpoints[B]
    endpoints[Asegment[0]] = endpoints[Asegment[-1]] = Asegment
    return Asegment

def greedy_tsp(cities):
    endpoints = {c: [c] for c in cities} # A dict of {endpoint: segment}
    for (A,B) in shortest_edges_first(cities):
        if A in endpoints and B in endpoints and endpoints[A] != endpoints[B]:
            new_segment = join_endpoints(endpoints,A,B)
            if len(new_segment) == len(cities):
                return new_segment

def altered_greedy_tsp(cities):
    """Run greedy TSP algorithm, and alter the result by reversing segements."""
    return alter_tour(greedy_tsp(cities))

def visualize_greedy_tsp(cities,plot_sizes):
    """
    Go through edges, shortest first. Use edge to join segments if possible.
    Plot segements at specified sizes.
    """
    edges = shortest_edges_first(cities)
    endpoints = {c: [c] for c in cities}
    for (A,B) in edges:
        if A in endpoints and B in endpoints and endpoints[A] != endpoints[B]:
            new_segment = join_endpoints(endpoints,A,B)
            plot_segments(endpoints, plot_sizes, distance(A,B))
            if len(new_segment) == len(cities):
                return new_segment

# main loop
if __name__ == '__main__':
    #cities = USA_landmarks_map()
    args = get_args()
    cities = Random_cities_map(args.size)
    if args.map == "USA":
        cities = USA_landmarks_map()
    if args.map == "USA-big":
        cities = USA_big_map()

    #plot_tsp(greedy_tsp,cities)
    plot_tsp(altered_greedy_tsp,cities)
    #visualize_greedy_tsp(cities,(50,25,10,5,2,1))

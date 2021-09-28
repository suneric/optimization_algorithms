"""
Dynamic Programming Algorithm (Optimal, inefficient)
Held-Harp dynamic programming is an algorithm for finding optimal tours, not approximate ones, so it is not appropriate for large n.
But even in its simplest form, without any programming tricks, it can go quite a bit further than alltours.
Because, alltouts wastes a lot of time with permutations that can't possibly be optimal tours.
Key property:
Given a start citt A, an end city C, and a set of middle cities Bs, then out of all the possible segments that
starts in A, end in C, and go through all and only the cities in Bs, only the shortest of those segments could
ever be part of an optimal tour.
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
import urllib
from map import *
from utils import *

def segment_length(segment):
    """The total of distances between each pair of consecutive cities in the segment."""
    # same as tour_length, but without distance(tour[0], tour[-1])
    return sum(distance(segment[i],segment[i-1]) for i in range(1,len(segment)))

# the decorator @functools.lru_cache makes this a dynamic programming algorithm,
# which is a fancy name meaning that we cache the results of sub-computation because
# we will re-use them multiple times.
# @functools.lru_cache(None) (python 3.2+)
def shortest_segment(A, Bs, C):
    """The shortest segment starting at A, going through all Bs, and ending at C."""
    if not Bs:
        return [A,C]
    else:
        segments = [shortest_segment(A, Bs-{B}, B) + [C] for B in Bs]
        return min(segments, key=segment_length)

def dp_tsp(cities):
    """
    The Held-Karp shortest tour of this set of cities.
    For each end city C, find the shortest segement from A (the start) to C.
    Out of all these shortest segment, pick the one tat is the shortest tour.
    """
    A = first(cities)
    return shortest_tour(shortest_segment(A, cities-{A,C},C) for C in cities if C is not A)

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
    plot_tsp(dp_tsp,cities)

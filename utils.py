import argparse
import itertools
import matplotlib.pyplot as plt
import time
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default="Random")
    parser.add_argument('--size', type=int, default=5) # door width
    return parser.parse_args()

def first(collection):
    """Start iterating over collection, and return the first element."""
    return next(iter(collection))

def distance(A,B):
    """The distance between two points"""
    return abs(A - B)

def tour_length(tour):
    """A tour starts in one city, and then visits each of the other cities,
    before returning to the start city.
    A natural representation of a tour is a sequence of cities."""
    return sum(distance(tour[i],tour[i-1]) for i in range(len(tour)))

def shortest_tour(tours):
    return min(tours, key=tour_length)

def valid_tour(tour, cities):
    return set(tour) == set(cities) and len(tour) == len(cities)

def plot_lines(points, style='bo-'):
    print(points)
    "Plot lines to connect a series of points."
    plt.plot([p.x for p in points], [p.y for p in points], style)
    plt.axis('scaled')
    plt.axis('off')

def plot_tour(tour):
    start = tour[0]
    plot_lines(list(tour)+[start])
    plot_lines([start],'rs') # mark the start city with a red square

def plot_tsp(algorithm, cities):
    """
    Apply a TSP algorithm to cities, plot the resulting tour, and print information
    """
    t0 = time.clock()
    tour = algorithm(cities)
    t1 = time.clock()
    assert valid_tour(tour,cities)
    print("{} city tour with length {:.1f} in {:.3f} secs for {}".format(
        len(tour), tour_length(tour), t1-t0, algorithm.__name__))
    plot_tour(tour)
    plt.show()

def plot_segments(endpoints, plot_sizes, dist):
    """If the number of distinct segments is one of plot_sizes, then plot segements."""
    segments = set(map(tuple,endpoints.values()))
    if len(segments) in plot_sizes:
        for s in segments:
            plot_lines(s)
        print('{} segments, longest edge = {:.0f}'.format(len(segments),dist))
        plt.show()


def sample(population, k, seed=42):
    """Return a list of k elements sampled from population."""
    if k is None or k > len(population):
        return population
    random.seed(len(population)*k*seed)
    return random.sample(population, k)

def shortest_edges_first(cities):
    """Return all edges between distinct cities, sorted shortest first."""
    edges = [(A,B) for A in cities for B in cities if id(A) < id(B)]
    return sorted(edges, key=lambda edge: distance(*edge))

def plot_graph(graph):
    """Given a graph of the form {parent: [child...]}, plot the vertices adn edges."""
    vertices = {v for parent in graph for v in graph[parent]} | set(graph)
    edges = {(parent, child) for parent in graph for child in graph[parent]}
    for edge in edges:
        plot_lines(edge,'ro-')
    total_length = sum(distance(p,c) for (p,c) in edges)
    print('{} node graph of total length: {:.1f}'.format(len(vertices), total_length))

"""
A problem with Nearest Neighbors is outliers. It seems that picking up an outlier is sometimes
a good idea, but sometimes going directly to the nearest neighbor is a better idea. It is
difficult to make the choice between an outlier and a nearest neighbor while awe are constructing
a tour, because we don't have the context of the whole tour yet. One way we could try to imporve
a tour us by reversing a segement
"""
def reverse_segement_if_better(tour,i,j):
    """If reversing tour[i:j] would make the tour shorter, then do it."""
    A,B,C,D = tour[i-1],tour[i],tour[j-1],tour[j%len(tour)]
    if distance(A,B)+distance(C,D) > distance(A,C)+distance(B,D):
        tour[i:j]=reversed(tour[i:j])

def all_segments(N):
    return [(start,start+length) for length in range(N,2-1,-1) for start in range(N-length+1)]

def alter_tour(tour):
    original_length = tour_length(tour)
    for (start,end) in all_segments(len(tour)):
        reverse_segement_if_better(tour,start,end)
    if tour_length(tour) < original_length:
        return alter_tour(tour)
    return tour

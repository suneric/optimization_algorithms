"""
Minimum Spanning Tree Traversal Algorithm (Approximate, guaranteed no twice longer than optimal path)
Construct a Minimum Spanning Tree, then do a pre-order traversal.
That will give you a tour that is guaranteed to be no more than twice as long as the minimal tour.
A graph is a collection of vertices and edges
A vertex us a point (such a city)
An edge is link between two vertices. Edges have lengths.
A directed graph is a graph where the edges have a direction. We sat that the edge goes from the parent vertex to the child vertex
A tree is a directed graph in which there is one distinguished vertex called the root that has no parent; every other vertex has exactly one parent
A spanning tree (of a set of vertices) is a tree that contains all the vertices
A minimum spanning tree is a spanning tree with the smallest possible sum of edge lengths
A traversal of a tree is a way of visiting all the vertices in some order
A pre-order traversal means that you visit the root first, then do a pre-order traversal of each of the children
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
import urllib
from map import *
from utils import *


"""
How to create a minimum spanning tree (MST)
Prim's algorithm for creating a MST: List all the edges and sort them, shortest first. Initialize a tree to be single
root city. Now repeat the following until the tree contains all the cities: find the shortest edge that links a city (A)
that in th tree to a city (B) that is not yet in th tree, and annd B to the list of A's children in the tree.
"""
def shortest_usable_edge(edges, tree):
    """Find the shortest edge (A,B) where A is in tree and B is not."""
    (A,B) = first((A,B) for (A,B) in edges if (A in tree) ^ (B in tree)) # ^ is "xor"
    return (A,B) if (A in tree) else (B,A)

def mst(vertices):
    """
    Given a set of vertices, build a minumum spanning tree:
    a dict of the form {parent:[child...]},
    where parent and children are vertices, and the root of the tree is first(vertices).
    """
    tree = {first(vertices):[]}
    edges = shortest_edges_first(vertices)
    while len(tree) < len(vertices):
        (A,B) = shortest_usable_edge(edges,tree)
        tree[A].append(B)
        tree[B] = []
    return tree

def preorder_traversal(tree, root):
    """Traverse tree in pre-order, starting at root of tree."""
    result = [root]
    for child in tree.get(root, ()):
        result.extend(preorder_traversal(tree,child))
    return result

def mst_tsp(cities):
    """Create a minimum spanning tree and walk it in pre-order, omitting duplicates."""
    return preorder_traversal(mst(cities),first(cities))

def altered_mst_tsp(cities):
    return alter_tour(mst_tsp(cities))

# main loop
if __name__ == '__main__':
    #cities = USA_landmarks_map()
    args = get_args()
    cities = Random_cities_map(args.size)
    if args.map == "USA":
        cities = USA_landmarks_map()
    if args.map == "USA-big":
        cities = USA_big_map()

    plot_tsp(altered_mst_tsp,cities)

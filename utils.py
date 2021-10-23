import argparse
import itertools
import matplotlib.pyplot as plt
import time
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default="Random")
    parser.add_argument('--size', type=int, default=5)
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

def lines(text): return text.strip().splitlines()
def Coordinate_map(lines, delimiter=' ', lat_col=1, long_col=2, lat_scale=69, long_scale=-48):
    """ Make a set of Cities from an iterable of lines of text.
    Specify the column delimiter, and the zero-based column number of lat and long.
    Treat long/lat as a squre x/y grid, scaled by long_scale and lat_scale.
    Source can be a file object, or list of lines."""

    return frozenset(City(long_scale*float(row[long_col]), lat_scale*float(row[lat_col]))
                    for row in csv.reader(lines, delimiter=delimiter, skipinitialspace=True))

def Random_cities_map(n, width=900, height=600, seed=42):
    """
    Make a set of size cities, each with random coordinate_with a (widthxheight) rectangle.
    """
    random.seed(seed*n)
    return frozenset(City(random.randrange(width), random.randrange(height)) for c in range(n))

def USA_landmarks_map():
    return Coordinate_map(lines("""
            [TCL]  33.23   87.62  Tuscaloosa,AL
            [FLG]  35.13  111.67  Flagstaff,AZ
            [PHX]  33.43  112.02  Phoenix,AZ
            [PGA]  36.93  111.45  Page,AZ
            [TUS]  32.12  110.93  Tucson,AZ
            [LIT]  35.22   92.38  Little Rock,AR
            [SFO]  37.62  122.38  San Francisco,CA
            [LAX]  33.93  118.40  Los Angeles,CA
            [SAC]  38.52  121.50  Sacramento,CA
            [SAN]  32.73  117.17  San Diego,CA
            [SBP]  35.23  120.65  San Luis Obi,CA
            [EKA]  41.33  124.28  Eureka,CA
            [DEN]  39.75  104.87  Denver,CO
            [DCA]  38.85   77.04  Washington/Natl,DC
            [MIA]  25.82   80.28  Miami Intl,FL
            [TPA]  27.97   82.53  Tampa Intl,FL
            [JAX]  30.50   81.70  Jacksonville,FL
            [TLH]  30.38   84.37  Tallahassee,FL
            [ATL]  33.65   84.42  Atlanta,GA
            [BOI]  43.57  116.22  Boise,ID
            [CHI]  41.90   87.65  Chicago,IL
            [IND]  39.73   86.27  Indianapolis,IN
            [DSM]  41.53   93.65  Des Moines,IA
            [SUX]  42.40   96.38  Sioux City,IA
            [ICT]  37.65   97.43  Wichita,KS
            [LEX]  38.05   85.00  Lexington,KY
            [NEW]  30.03   90.03  New Orleans,LA
            [BOS]  42.37   71.03  Boston,MA
            [PWM]  43.65   70.32  Portland,ME
            [BGR]  44.80   68.82  Bangor,ME
            [CAR]  46.87   68.02  Caribou Mun,ME
            [DET]  42.42   83.02  Detroit,MI
            [STC]  45.55   94.07  St Cloud,MN
            [DLH]  46.83   92.18  Duluth,MN
            [STL]  38.75   90.37  St Louis,MO
            [JAN]  32.32   90.08  Jackson,MS
            [BIL]  45.80  108.53  Billings,MT
            [BTM]  45.95  112.50  Butte,MT
            [RDU]  35.87   78.78  Raleigh-Durh,NC
            [INT]  36.13   80.23  Winston-Salem,NC
            [OMA]  41.30   95.90  Omaha/Eppley,NE
            [LAS]  36.08  115.17  Las Vegas,NV
            [RNO]  39.50  119.78  Reno,NV
            [AWH]  41.33  116.25  Wildhorse,NV
            [EWR]  40.70   74.17  Newark Intl,NJ
            [SAF]  35.62  106.08  Santa Fe,NM
            [NYC]  40.77   73.98  New York,NY
            [BUF]  42.93   78.73  Buffalo,NY
            [ALB]  42.75   73.80  Albany,NY
            [FAR]  46.90   96.80  Fargo,ND
            [BIS]  46.77  100.75  Bismarck,ND
            [CVG]  39.05   84.67  Cincinnati,OH
            [CLE]  41.42   81.87  Cleveland,OH
            [OKC]  35.40   97.60  Oklahoma Cty,OK
            [PDX]  45.60  122.60  Portland,OR
            [MFR]  42.37  122.87  Medford,OR
            [AGC]  40.35   79.93  Pittsburgh,PA
            [PVD]  41.73   71.43  Providence,RI
            [CHS]  32.90   80.03  Charleston,SC
            [RAP]  44.05  103.07  Rapid City,SD
            [FSD]  43.58   96.73  Sioux Falls,SD
            [MEM]  35.05   90.00  Memphis Intl,TN
            [TYS]  35.82   83.98  Knoxville,TN
            [CRP]  27.77   97.50  Corpus Chrst,TX
            [DRT]  29.37  100.92  Del Rio,TX
            [IAH]  29.97   95.35  Houston,TX
            [SAT]  29.53   98.47  San Antonio,TX
            [LGU]  41.78  111.85  Logan,UT
            [SLC]  40.78  111.97  Salt Lake Ct,UT
            [SGU]  37.08  113.60  Saint George,UT
            [CNY]  38.77  109.75  Moab,UT
            [MPV]  44.20   72.57  Montpelier,VT
            [RIC]  37.50   77.33  Richmond,VA
            [BLI]  48.80  122.53  Bellingham,WA
            [SEA]  47.45  122.30  Seattle,WA
            [ALW]  46.10  118.28  Walla Walla,WA
            [GRB]  44.48   88.13  Green Bay,WI
            [MKE]  42.95   87.90  Milwaukee,WI
            [CYS]  41.15  104.82  Cheyenne,WY
            [SHR]  44.77  106.97  Sheridan,WY
            """))

def continental_USA(line):
    return line.startswith('[') and ',AK' not in line and ',HI' not in line

"""
Data for tsp
https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html
"""
def USA_big_map():
    url = 'http://www.realestate3d.com/gps/latlong.htm'
    return Coordinate_map(filter(continental_USA, urllib.urlopen(url)))

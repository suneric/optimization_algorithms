"""
discrete coverage path planning problem
it could be divided into two problems: first, solving the set covering problem to find a minimum ser of
viewpoints among a bounch of candidate viewpoints that fully cover the area of interest; second, solving
the traverling salesman problem to find a shortest visit path among the set of selected viewpoints.
"""
import time
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from map import *
from utils import *
from scp_greedy import *
from tsp_aco import *
import os

def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--mapfile', type=str, default=None)
    parser.add_argument('--vpsfile', type=str, default=None)
    parser.add_argument('--trajfile', type=str, default=None)

    parser.add_argument('--ants', type=int, default=10)
    parser.add_argument('--maxIter', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--rho', type=float, default=0.5)
    return parser.parse_args()

# main loop
if __name__ == '__main__':
    args = getParameters()

    vps = []
    map = GridMap()
    width, height = 0, 0
    if args.load:
        map.loadMap(os.path.join(args.load, args.mapfile))
        width, height = map.width, map.height
        vps = loadViewpoints(os.path.join(args.load, args.vpsfile),map)

    # solving covering set problem with greedy
    startIdx = 0#np.random.randint(len(vps))
    minvps = computeMinimumCoveringViewpoints(map=map, vps=vps, startIdx=startIdx)

    # use ACO to solve TSP
    cities = [City(vp.location[0], vp.location[1]) for vp in minvps]
    tspACO = ACO(cities=list(cities), ants=args.ants, maxIter=args.maxIter, alpha=args.alpha, beta=args.beta, rho=args.rho)
    progress, bestTour = tspACO.run()

    # rotate list
    idx = bestTour.index(cities[startIdx])
    bestTour = bestTour[idx:]+bestTour[:idx]

    # save trajectory
    bestVps = [minvps[cities.index(city)] for city in bestTour]
    saveViewpoints(os.path.join(args.load, args.trajfile), bestVps)

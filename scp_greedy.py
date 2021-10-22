import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from map import *
from utils import *
import time
import copy

"""
plot viewpoints position
"""
def plotViewPoints(ax, viewPts, plotCoverage=True, plotBoundary=False):
    ptx, pty = [], []
    for vp in viewPts:
        ptx.append(vp.location[0])
        pty.append(vp.location[1])
        if plotCoverage:
            vp.plotView(ax,plotBoundary)

    ax.scatter(ptx, pty, s=3, c='r', marker='*')


"""
Greedy Set Covering
The main idea of the algorithm is to cover the universe taking every time the apprently most
convenient set in the sets lists. In other words, every while cycle the program will search among
all sets and will take the one with the highest ratio bwteen the elements not yet covered and the
ralative cost of the set. This algorithm doesn't always give the best result, but certainly it
gives an optimal one.

"""
def set_cover(universe, subsets, costs, startIdx):
    cost = 0
    elements = set(e for s in subsets for e in s)
    # check full coverage
    if elements != universe:
        print("not fully covered:",len(elements), len(universe))
        return None

    covered = set()
    cover = []

    start = subsets[startIdx]
    cover.append(start)
    cost += costs[startIdx]
    covered |= start
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s-covered)/costs[subsets.index(s)])
        cover.append(subset)
        cost += costs[subsets.index(subset)]
        covered |= subset # union two sets

    print("covered size {}, universe size {}".format(len(covered), len(universe)))
    return cover, cost

"""
Compute an approximate minimum set of viewpoints to cover the who valid grid
"""
def computeMinimumCoveringViewpoints(map,vps,startIdx):
    universe = set([grid.id for grid in map.landGrids])
    subsets = []
    costs = []
    for i in range(len(vps)):
        vpset = set([grid.id for grid in vps[i].landCover])
        subsets.append(vpset)
        costs.append(1.0) # same cost for every viewpoint

    t0 = time.clock()
    cover, cost = set_cover(universe,subsets,costs,startIdx)
    t1 = time.clock()

    # duplicate grid
    duplicate1, duplicate2 = computeDuplication(universe,cover)
    print("Find {} viewpoints in {} candidates in {:.3f} secs with cost {:.3f} and duplication {:.3f} {:.3f}".format(len(cover), len(vps), t1-t0, cost, duplicate1, duplicate2))

    # return the view points
    minvps = []
    for s in cover:
        index = subsets.index(s)
        minvps.append(vps[index])

    return minvps

"""
duplicate1: minimum duplicaitoin with count overlap grid only once
duplicate2: maximum duplicatioin with count grid whenever it overlap with another
"""
def computeDuplication(universe, cover):
    intersectSet = set()
    for s1 in cover:
        for s2 in cover:
            if s1 != s2:
                s = s1 & s2 #interestion
                for e in s:
                    intersectSet.add(e)
    #print(intersectSet)
    totalLen = float(len(universe))
    duplicate1 = float(len(intersectSet))/totalLen
    duplicate2 = float(sum([len(subset) for subset in cover]) - totalLen)/totalLen
    return duplicate1, duplicate2



def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapwidth', type=int, default=100)
    parser.add_argument('--mapheight', type=int, default=100)
    parser.add_argument('--mapres', type=float, default=1.0) # grid resolution
    parser.add_argument('--mapseeds', type=int, default=1000) # number of seeds
    parser.add_argument('--viewdis',type=float,default=5.0) # working distance
    parser.add_argument('--vpmethod', type=str, default="uniform") # viewpoints generation method
    return parser.parse_args()

# main loop
if __name__ == '__main__':
    args = getArgs()

    fig = plt.figure(figsize=(args.mapwidth/5,args.mapheight/5))
    ax = fig.add_subplot(111)

    # generate grid map
    map = GridMap(width=args.mapwidth,height=args.mapheight,res=args.mapres,n=args.mapseeds)
    map.plotMap(ax)
    # generate viewpoints
    vps = generateViewPoints(gridMap = map,fov = (80.0,80.0), workingDistance = args.viewdis, type = args.vpmethod)

    # solving covering set problem with greedy
    minvps = computeMinimumCoveringViewpoints(map = map, viewPts = vps)
    plotViewPoints(ax = ax, viewPts = minvps,plotCoverage = True, plotBoundary=True)

    plt.show()

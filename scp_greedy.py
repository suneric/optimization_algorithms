import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from map import *
from utils import *
import time


"""
Generate viewpoints
gridMap: the map with grid information
workingDistance: the distance from the ground to the camera
type: "random": randomly generated the viewpoints, "uniform": each grid will have a viewpoint above on it.
return a list of viewpoints
"""
def generateViewPoints(gridMap, fov, workingDistance = 3, type = "uniform"):
    vps = []
    size = len(gridMap.grids)
    for c in range(size):
        base = (0,0)
        if type == "random":
            base = (random.randrange(gridMap.width), random.randrange(gridMap.height))
        else:
            base = gridMap.grids[c].center()

        # create a viewpoint with given distance and rotation angle 0.0
        pose = (base[0], base[1], workingDistance, 0.0)
        vp = ViewPoint(location=pose, fov = fov, id=c)
        vps.append(vp)

    for vp in vps:
        computeViewCoverGrids(gridMap.grids, vp)

    return vps

"""
compute the covering grid under a viewpoint
"""
def computeViewCoverGrids(grids, viewPt):
    xmin,xmax,ymin,ymax = viewPt.view[0][0],viewPt.view[1][0],viewPt.view[1][1],viewPt.view[2][1]
    for grid in grids:
        if grid.anchor[0] > xmax or grid.anchor[0]+grid.length < xmin:
            continue
        if grid.anchor[1] > ymax or grid.anchor[1]+grid.length < ymin:
            continue
        # only count the valid grid
        if grid.status == 1:
            viewPt.cover.append(grid)
    return viewPt.cover

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
def set_cover(universe, subsets, costs):
    cost = 0
    elements = set(e for s in subsets for e in s)
    # check full coverage
    if elements != universe:
        print("not fully covered:",len(elements), len(universe))
        return None

    covered = set()
    cover = []
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
def computeMinimumCoveringViewpoints(map,viewPts):
    validGrids = []
    for grid in map.grids:
        if grid.status == 1:
            validGrids.append(grid.id)

    universe = set(validGrids)
    subsets = []
    costs = []
    for i in range(len(viewPts)):
        vpset = set([grid.id for grid in viewPts[i].cover])
        #print(vpset)
        subsets.append(vpset)
        costs.append(1.0) # same cost for every viewpoint

    t0 = time.clock()
    cover, cost = set_cover(universe,subsets,costs)
    t1 = time.clock()

    # duplicate grid
    duplicate1, duplicate2 = computeDuplication(universe,cover)
    print("Find {} viewpoints in {} candidates in {:.3f} secs with cost {:.3f} and duplication {:.3f} {:.3f}".format(len(cover), len(viewPts), t1-t0, cost, duplicate1, duplicate2))

    # return the view points
    minVps = []
    for s in cover:
        index = subsets.index(s)
        minVps.append(viewPts[index])

    return minVps

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
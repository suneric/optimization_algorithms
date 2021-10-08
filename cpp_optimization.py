"""
discrete coverage path planning problem

approaches

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

def plot_lines(ax, points, style='bo-',width=2, markersize=10, addArrow=False):
    "Plot lines to connect a series of points."
    X,Y = [p.x for p in points],[p.y for p in points]
    EX,EY=[(points[i+1].x-points[i].x)/2 for i in range(len(points)-1)], [(points[i+1].y-points[i].y)/2 for i in range(len(points)-1)]
    ax.plot(X,Y,style,linewidth=width,markersize=markersize)
    if addArrow:
        ax.quiver(X,Y,EX,EY,color='b', units='xy', scale=1, width=0.5)
    ax.axis('scaled')
    ax.axis('off')

def plotTrajectory(ax,tour,addArrow=False):
    start = tour[0]
    plot_lines(ax,list(tour)+[start],addArrow=addArrow)
    plot_lines(ax,[start],'yh',markersize=20) # mark the start city with a red square

def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapwidth', type=int, default=100)
    parser.add_argument('--mapheight', type=int, default=100)
    parser.add_argument('--mapres', type=float, default=1.0) # grid resolution
    parser.add_argument('--mapseeds', type=int, default=1000) # number of seeds
    parser.add_argument('--viewdis',type=float,default=5.0) # working distance
    parser.add_argument('--vpmethod', type=str, default="uniform") # viewpoints generation method

    parser.add_argument('--ants', type=int, default=10)
    parser.add_argument('--maxIter', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--rho', type=float, default=0.5)
    return parser.parse_args()

# main loop
if __name__ == '__main__':
    args = getParameters()

    fig1 = plt.figure(figsize=(args.mapwidth/5,args.mapheight/5))
    ax = fig1.add_subplot(111)

    # generate grid map
    map = GridMap(width=args.mapwidth,height=args.mapheight,res=args.mapres,n=args.mapseeds)
    map.plotMap(ax)

    # generate viewpoints
    vps = generateViewPoints(gridMap = map,fov = (80.0,80.0), workingDistance = args.viewdis, type = args.vpmethod)
    #plotViewPoints(ax = ax, viewPts = minvps,plotCoverage = True, plotBoundary=False)

    # solving covering set problem with greedy
    minvps = computeMinimumCoveringViewpoints(map = map, viewPts = vps)
    plotViewPoints(ax = ax, viewPts = minvps,plotCoverage = True, plotBoundary=False)

    # use ACO to solve TSP
    cities = frozenset(City(vp.location[0], vp.location[1]) for vp in minvps)
    tspACO = ACO(cities = list(cities), ants = args.ants, maxIter = args.maxIter, alpha = args.alpha, beta = args.beta, rho = args.rho)
    progress, bestTour = tspACO.run()
    plotTrajectory(ax, bestTour, addArrow=False)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(progress)
    ax2.set_ylabel('Distance')
    ax2.set_xlabel('Iteration')

    plt.show()

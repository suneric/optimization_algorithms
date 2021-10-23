"""
discrete coverage path planning problem
with monte carlo tree search algorithm
"""
import time
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from map import *
from utils import *
import sys
import os
from cpp_mcts import MCTSUtil

############################################################
# main
def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapwidth', type=int, default=100)
    parser.add_argument('--mapheight', type=int, default=100)
    parser.add_argument('--mapres', type=float, default=1.0) # grid resolution
    parser.add_argument('--mapseeds', type=int, default=1000) # number of seeds
    parser.add_argument('--viewdis',type=float,default=5.0) # working distance
    parser.add_argument('--vpres', type=float ,default=2.0)
    parser.add_argument('--vpmethod', type=str, default="uniform") # viewpoints generation method

    parser.add_argument('--nb', type=int, default=8)
    parser.add_argument('--cn', type=float, default=0.3)

    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--mapfile', type=str, default=None)
    parser.add_argument('--vpsfile', type=str, default=None)
    parser.add_argument('--draw', type=int, default=0) # 0 no draw, 1, draw map, 2, draw map and ViewPoints, 3, draw map and trajectory
    return parser.parse_args()

if __name__ == "__main__":
    args = getParameters()
    # create map, viewpoints and save
    vps = []
    map = GridMap()
    if args.save:
        map.makeMap(width=args.mapwidth,height=args.mapheight,res=args.mapres,sn=args.mapseeds)
        map.saveMap(os.path.join(args.save, args.mapfile))
        vps = generateViewPoints(gridMap = map,fov = (80.0,80.0), resolution=args.vpres, workingDistance = args.viewdis, type = args.vpmethod)
        saveViewpoints(os.path.join(args.save, args.vpsfile),vps)

    if args.load:
        map.loadMap(os.path.join(args.load, args.mapfile))
        vps = loadViewpoints(os.path.join(args.load, args.vpsfile),map)

    fig = plt.figure(figsize=(15,12)) # inch
    ax = fig.add_subplot(111)

    if args.draw == 1: # map only
        map.plotMap(ax)
    elif args.draw == 2: # map and viewpoints
        map.plotMap(ax)
        plotViewpoints(ax,vps,type=0)
    elif args.draw == 3: # map and trajectory
        cities = [City(vp.location[0], vp.location[1]) for vp in vps]
        drawTrajectory(ax,map,cities,vps,speed=10)
    elif args.draw == 4: # draw neighbor
        startIdx = np.random.randint(len(vps))
        startVp = vps[startIdx]
        util = MCTSUtil(map, vps, actDim=args.nb, cn=args.cn)
        nbvps = [vps[i] for i in util.neighbors(startVp.id)[0:args.nb]]
        nbvps.insert(0,startVp)
        cities = [City(vp.location[0], vp.location[1]) for vp in nbvps]
        drawTrajectory(ax,map,cities,nbvps,speed=10,drawline=False)
    plt.show()

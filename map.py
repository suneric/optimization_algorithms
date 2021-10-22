#
import random
import csv
import urllib
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from utils import *
import sys
import os

random.seed(44)

"""
SquareGrid
anchor: the base point of the square
length: side length of the square
id: unique id
status: 0 or 1, for the grid is valid on map
cover: covered by a view: 0, no covering, n covered by n views
"""
class SquareGrid:
    def __init__(self, anchor, length, id):
        self.anchor = anchor # bottom left position [width, height]
        self.length = length
        self.id = id
        self.status = 0 # 0: empty (water), 1: valid (land)

    # check in a pt is in grid
    def inGrid(self, pt):
        if pt[0] < self.anchor[0] or pt[0] > self.anchor[0]+self.length:
            return False
        if pt[1] < self.anchor[1] or pt[1] > self.anchor[1]+self.length:
            return False
        return True

    def center(self):
        return (self.anchor[0]+0.5*self.length, self.anchor[1]+0.5*self.length)

"""
GridMap
"""
class GridMap:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.grids = []
        self.landGrids = []
        self.seeds = []

    # make a map with width of 90 meters and height of 60 meters
    def makeMap(self, width = 90, height = 60, res=1, sn=1000):
        self.height = height
        self.width = width
        self.seeds = np.array([(random.randrange(width), random.randrange(height)) for c in range(sn)])
        self.grids = self.makeGrids(res)
        self.landGrids = self.makeLands()
        print("make map ({} meters x {} meters) width {} grids in total, covering {} land grids.".format(self.width, self.height, len(self.grids), len(self.landGrids)))

    def loadMap(self,filename):
        with open(filename, 'r') as reader:
            lines = reader.read().splitlines()
            for i in range(len(lines)):
                data = lines[i].split(" ")
                if i == 0:
                    self.height = float(data[0])
                    self.width = float(data[1])
                else:
                    id = int(data[0])
                    anchor = [float(data[1]),float(data[2])]
                    length = float(data[3])
                    grid = SquareGrid(anchor,length,id)
                    grid.status = int(data[4])
                    self.grids.append(grid)
                    if grid.status == 1:
                        self.landGrids.append(grid)
        reader.close()
        print("load map ({} meters x {} meters) width {} grids in total, covering {} land grids.".format(self.width, self.height, len(self.grids), len(self.landGrids)))

    def saveMap(self,filename):
        with open(filename,'w') as writer:
            writer.write(str(self.height) + " " + str(self.width) + "\n")
            for i in range(len(self.grids)):
                grid = self.grids[i]
                line = str(grid.id) + " "\
                     + str(grid.anchor[0]) + " "\
                     + str(grid.anchor[1]) + " "\
                     + str(grid.length) + " "\
                     + str(grid.status) + "\n"
                writer.write(line)
            writer.close()

    def makeGrids(self, res=1):
        grids = []
        nrow = int(self.height/res)
        ncol = int(self.width/res)
        for i in range(nrow):
            for j in range(ncol):
                sg = SquareGrid(anchor=(j*res,i*res), length=res, id=j+i*ncol)
                grids.append(sg)
        return grids

    def makeLands(self):
        lands = []
        for grid in self.grids:
            if self.checkStatus(grid):
                lands.append(grid)
        return lands

    def checkStatus(self,grid):
        if not grid.status:
            for pt in self.seeds:
                if grid.inGrid(pt):
                    grid.status = 1 # occupied by seed
                    break
        return grid.status

    def plotMap(self, ax, plotSeeds = False):
        ax.autoscale(enable=False)
        ax.set_xlim([-10,self.width+10])
        ax.set_ylim([-10,self.height+10])

        if plotSeeds:
            ax.scatter(self.seeds[:,0], self.seeds[:,1], s=1, c='g', marker='o')

        for grid in self.landGrids:
            patch = matplotlib.patches.Rectangle((grid.anchor),grid.length, grid.length, facecolor = "green", edgecolor='black',linewidth=1.0,alpha=0.2)
            ax.add_patch(patch)
            #ax.text(grid.anchor[0],grid.anchor[1],str(grid.id))

"""
A ViewPoint is identifed with its location (x,y,z,yaw) and the sensor FOV(angle1,angle2)
yaw (about z axis), angle1 (in x direction) an angle2 (in y direction) are measured in degree
"""
class ViewPoint:
    def __init__(self,location=(0.0,0.0,1.0,90.0),fov=(60.0,60.0),id=0):
        self.location = location
        self.fov = fov
        self.id = id
        self.view = self.coverArea()
        self.gridCover = []
        self.landCover = []
        self.waterCover = []

    def coverArea(self):
        """
        cover area is calculated with give the working distance (z) and the FOV
        return a rectangle vertices in np.array [x,y,0]
        """
        center = (self.location[0],self.location[1])
        fov1 = math.radians(0.5*self.fov[0])
        fov2 = math.radians(0.5*self.fov[1])
        xlen = self.location[2]*np.tan(fov1)
        ylen = self.location[2]*np.tan(fov2)
        xmin = center[0] - xlen
        xmax = center[0] + xlen
        ymin = center[1] - ylen
        ymax = center[1] + ylen

        yaw = math.radians(self.location[3])
        rmat = np.matrix([[np.cos(yaw),-np.sin(yaw),0],
                          [np.sin(yaw),np.cos(yaw),0],
                          [0,0,1]])

        pt0 = self.rotatePoint(center,yaw,(xmin,ymin))
        pt1 = self.rotatePoint(center,yaw,(xmax,ymin))
        pt2 = self.rotatePoint(center,yaw,(xmax,ymax))
        pt3 = self.rotatePoint(center,yaw,(xmin,ymax))
        return (pt0,pt1,pt2,pt3)

    def rotatePoint(self,center,angle,p):
        s = np.sin(angle)
        c = np.cos(angle)
        x = p[0]-center[0]
        y = p[1]-center[1]
        xnew = x*c-y*s + center[0]
        ynew = x*s+y*c + center[1]
        return (xnew,ynew)

    def plotView(self,ax,type=0):
        if type == 0: # viewpoint
            x,y = self.location[0], self.location[1]
            ax.scatter([x],[y], s=1, c='red', marker='o')
        elif type == 1: # boundary
            x = [self.view[0][0],self.view[1][0],self.view[2][0],self.view[3][0],self.view[0][0]]
            y = [self.view[0][1],self.view[1][1],self.view[2][1],self.view[3][1],self.view[0][1]]
            ax.plot(x,y,linewidth=2,color="red")
        else: # coverage
            for grid in self.landCover:
                patch = matplotlib.patches.Rectangle((grid.anchor),grid.length, grid.length, facecolor = "red", edgecolor='black',linewidth=1.0,alpha=0.2)
                ax.add_patch(patch)



"""
Generate viewpoints
gridMap: the map with grid information
workingDistance: the distance from the ground to the camera
type: "random": randomly generated the viewpoints, "uniform": each grid will have a viewpoint above on it.
return a list of viewpoints
"""
def generateViewPoints(gridMap, fov, resolution = 2.0, workingDistance = 3, type = "uniform"):
    vps = []
    grids = gridMap.makeGrids(resolution) # make another grids from the map for generating viewpoint
    size = len(grids)
    for c in range(size):
        base = (0,0)
        if type == "random":
            base = (random.randrange(gridMap.width), random.randrange(gridMap.height))
        else:
            base = grids[c].center()

        # create a viewpoint with given distance and rotation angle 0.0
        pose = (base[0], base[1], workingDistance, 0.0)
        vp = ViewPoint(location=pose, fov = fov, id=c)
        vps.append(vp)

    for vp in vps:
        computeViewCoverGrids(gridMap.grids, vp)

    print("generate {} viewpoints with resolution {:.2f} at working distance {:.2f}".format(len(vps), resolution, workingDistance))
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

        viewPt.gridCover.append(grid)
        if grid.status == 1:
            viewPt.landCover.append(grid)
        else:
            viewPt.waterCover.append(grid)
    return


def loadViewpoints(filename, map):
    vps = []
    with open(filename,'r') as reader:
        for line in reader.read().splitlines():
            sections = line.split(",")
            #print(sections)
            data = sections[0].split(" ")
            id = int(data[0])
            location = (float(data[1]), float(data[2]), float(data[3]), float(data[4]))
            fov = (float(data[5]), float(data[6]))
            vp = ViewPoint(location,fov,id)

            gridIndices = sections[1].split(" ")
            for id in gridIndices:
                vp.gridCover.append(map.grids[int(id)])
            landIndices = sections[2].split(" ")
            for id in landIndices:
                vp.landCover.append(map.grids[int(id)])
            if len(sections) > 3:
                waterIndices = sections[3].split(" ")
                if len(waterIndices) > 0:
                    for id in waterIndices:
                        vp.waterCover.append(map.grids[int(id)])
            vps.append(vp)
    reader.close()
    print("load {} viewpoints".format(len(vps)))
    return vps

def saveViewpoints(filename, vps):
    with open(filename, 'w') as writer:
        for vp in vps:
            line = str(vp.id) + " "\
                 + str(vp.location[0]) + " " + str(vp.location[1]) + " " + str(vp.location[2]) + " " + str(vp.location[3]) + " "\
                 + str(vp.fov[0]) + " " + str(vp.fov[1]) + ","
            for i in range(len(vp.gridCover)-1):
                line += str(vp.gridCover[i].id) + " "
            line += str(vp.gridCover[len(vp.gridCover)-1].id)
            line +=","
            for i in range(len(vp.landCover)-1):
                line += str(vp.landCover[i].id) + " "
            line += str(vp.landCover[len(vp.landCover)-1].id)
            if len(vp.waterCover) > 0:
                line +=","
                for i in range(len(vp.waterCover)-1):
                    line += str(vp.waterCover[i].id) + " "
                line += str(vp.waterCover[len(vp.waterCover)-1].id)
            line +="\n"
            writer.write(line)
    writer.close()
    return

def plotViewpoints(ax, vps, type=0):
    for vp in vps:
        vp.plotView(ax,type)
    return


"""
Plot the line and trajectory
"""
def vpDistance(vp1,vp2):
    dx = vp1.location[0]-vp2.location[0]
    dy = vp1.location[1]-vp2.location[1]
    dz = vp1.location[2]-vp2.location[2]
    return math.sqrt(dx*dx+dy*dy+dz*dz)

def vpLandOverlapCount(vp1,vp2):
    set1 = set(vp1.landCover)
    set2 = set(vp2.landCover)
    intersect = set1 & set2
    return len(intersect)


"""
Plot the line and trajectory
"""
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

"""
Draw the trajectory and coverage dynamically with a specified speed
"""
def drawTrajectory(ax,map,tour,vps,speed=3,drawline=True):
    # draw map
    map.plotMap(ax)
    plt.draw()
    # draw start viewpoint
    start = tour[0]
    plot_lines(ax,[start],'bh',markersize=16)
    vps[0].plotView(ax,type=2)
    plt.draw()
    plt.pause(1) # puase 1 second to start
    # draw trajectory
    totalDist = 0
    for i in range(1, len(tour)):
        next = tour[i]
        if drawline:
            plot_lines(ax,[start,next], markersize=8)
        else:
            plot_lines(ax,[next], markersize=8)
        vps[i].plotView(ax,type=2)
        plt.draw()
        dist = distance(start,next)
        totalDist += dist
        plt.pause(dist/speed)
        start = next

    # return to start
    # plot_lines(ax,[start,tour[0]])
    # dist = distance(start,tour[0])
    # totalDist += dist
    # plt.draw()

    #print("The best solution visits {} viewpoints, resulting in {:.3f} meters of total traveling distance.".format(len(tour),totalDist))
    plt.text(0,-5,"Done with {:.2f} meters traveling {} viewpoints.".format(totalDist,len(tour)))

"""
Point indicate the location of city
"""
class Point(complex):
    x = property(lambda p: p.real)
    y = property(lambda p: p.imag)

City = Point

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

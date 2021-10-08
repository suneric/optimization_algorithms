#
import random
import csv
import urllib
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

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
        self.cover = 0 # not covered

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
    # make a map with width of 90 meters and height of 60 meters
    def __init__(self, width = 90, height = 60, res=1, n=1000, seed=11):
        self.height = height
        self.width = width
        random.seed(seed*n)
        self.seeds = np.array([(random.randrange(width), random.randrange(height)) for c in range(n)])
        self.grids = self.makeGrids(res)
        self.ratio = self.computeLandRatio(self.grids)

    def computeLandRatio(self,grids):
        land = 0
        for grid in grids:
            if grid.status:
                land += 1
        ratio = float(land) / float(len(grids))
        print("{} valid grids in total {} grids with valid ratio {:.3f} ".format(land, len(grids), ratio))
        return ratio

    def makeGrids(self, res=1):
        nrow = int(self.height/res)
        ncol = int(self.width/res)
        grids = []
        for i in range(nrow):
            for j in range(ncol):
                sg = SquareGrid(anchor=(j*res,i*res), length=res, id=j+i*ncol)
                grids.append(sg)
        print("Generate {} grids with resolution of {} meters".format(len(grids), res))

        # update status
        for grid in grids:
            self.checkStatus(grid)
            #print(grid.anchor,grid.status)

        return np.array(grids)

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

        for grid in self.grids:
            if grid.status:
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
        self.cover = []

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

    def plotView(self,ax,plotBoundary=False):
        if plotBoundary:
            x = [self.view[0][0],self.view[1][0],self.view[2][0],self.view[3][0],self.view[0][0]]
            y = [self.view[0][1],self.view[1][1],self.view[2][1],self.view[3][1],self.view[0][1]]
            ax.plot(x,y,linewidth=2,color="red")

        for grid in self.cover:
            if grid.status:
                patch = matplotlib.patches.Rectangle((grid.anchor),grid.length, grid.length, facecolor = "red", edgecolor='black',linewidth=1.0,alpha=0.2)
                ax.add_patch(patch)

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

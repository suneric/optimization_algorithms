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
import time
import math
import copy

MAXVALUE = 1000000

class MCTSUtil(object):
    def __init__(self, map, viewpoints, actDim=8,cn=0.3):
        self.map = map
        self.actDim = actDim
        self.viewpoints = viewpoints
        self.nbMap = self.buildNeighborMap(viewpoints,cn)

    def neighbors(self, vpIdx):
        return self.nbMap[vpIdx]

    """
    build a map for viewpoints
    """
    def buildNeighborMap(self,vps,cn):
        print("=== start building neighbor map ===".format())
        dim = len(vps)
        nbMap = [None]*dim
        for i in range(dim):
            # print("{} / {} viewpoint".format(i+1,dim))
            nbvps = self.searchNeighbor(i,vps,cn)
            nbMap[i] = nbvps
        print("=== end building neighbor map ===".format())
        return nbMap

    """
    search neighbor viewpoints of a given viewpoint
    considering the distance and overlap
    """
    def searchNeighbor(self,i,vps,cn):
        dim = len(vps)
        vp = vps[i]
        scoreList = [None]*dim
        for j in range(dim):
            vp1 = vps[j]
            if vp1.id == vp.id:
                scoreList[j] = MAXVALUE
            else:
                dist = vpDistance(vp, vp1)
                overlap = vpLandOverlapCount(vp, vp1)
                scoreList[j]=cn*dist + (1.0-cn)*overlap
        # return the viewpoint indices with least value
        sortedIndice = np.argsort(np.array(scoreList))
        # sorted = np.lexsort((range(dim), scoreList))[0:self.actDim]
        return sortedIndice

    """
    reset / initial state
    """
    def initialState(self, startVp):
        vpsState = [0]*len(self.viewpoints)
        vpsState[startVp.id] = 1
        gridState = [0]*len(self.map.grids)
        for grid in startVp.landCover:
            gridState[grid.id] = 1

        coveredGrids = np.asarray(np.nonzero(gridState)).flatten()
        coverage = float(len(coveredGrids))/float(len(self.map.landGrids)) # only count land coverage
        # print("initial state", startVp.id, coverage, len(self.map.landGrids))
        state = MCTSState(self, startVp, vpsState, gridState, coverage, 0, 0.0)
        return state


"""
State represent the visited viewpoints and covered grids
"""
class MCTSState(object):
    def __init__(self, util, currVp, vpsState, gridState, coverage, overlap, traveledDist=0.0):
        self.util = util
        self.currVp = currVp
        self.vpsState = vpsState
        self.gridState = gridState
        self.coverage = coverage # total coverage
        self.overlap = overlap
        self.traveledDist = traveledDist # total distance
        self.nbvps = self.util.neighbors(currVp.id)

    def isGameOver(self):
        return self.coverage == 1.0

    def score(self):
        """
        return a score indicating how good the state is
        considering coverage, overlap and traveled distance
        """
        return 100*self.coverage - 0.1*self.overlap - 0.01*self.traveledDist

    def neighbors(self):
        """
        return neighbor viewpoints which have not been visited
        """
        unvisited = []
        for i in range(len(self.nbvps)-1): # ignore the last one which is itself
            if len(unvisited) >= self.util.actDim:
                break
            else:
                vpIdx = self.nbvps[i]
                if not self.vpsState[vpIdx]:
                    unvisited.append(vpIdx)
        return unvisited

    def move(self, nextVp):
        """
        move to next vp and return a new state
        """
        # copy viewpoints and grids tates
        vpsState = copy.deepcopy(self.vpsState)
        gridState = copy.deepcopy(self.gridState)

        # update  viewpoints and grids states
        vpsState[nextVp.id] = 1
        viewGrids = nextVp.landCover
        overlap = 0
        for grid in viewGrids:
            if gridState[grid.id]:
                overlap += 1
            else:
                gridState[grid.id] = 1

        coveredGrids = np.asarray(np.nonzero(gridState)).flatten()
        coverage = float(len(coveredGrids))/float(len(self.util.map.landGrids)) # only count land coverage
        dist = self.traveledDist + vpDistance(self.currVp, nextVp)
        return MCTSState(self.util, nextVp, vpsState, gridState, coverage, overlap, dist)


# base Node
class MCTSNode(object):
    def __init__(self, util, state, parent=None):
        self.util = util
        self.state = state
        self.untriedVps = self.state.neighbors()

        self.totalReward = 0.
        self.numberOfVisit = 0.
        self.maxReward = 0.

        self.parent = parent
        self.children = []

    def viewpoint(self):
        """
        corresponding viewpoint
        """
        return self.state.currVp

    def isTerminalNode(self):
        return self.state.isGameOver()

    def isFullyExpanded(self):
        return len(self.untriedVps) == 0

    def best_child(self, c, e):
        """
        return a child node with control parameter and eplison
        """
        if np.random.rand() > e:
            weights = [child.q(c) for child in self.children]
            return self.children[np.argmax(weights)]
        else:
            return self.children[np.random.randint(len(self.children))]

    def q(self,c):
        """
        q value of node based on max reward, average reward
        """
        return c*self.maxReward + (1-c)*(self.totalReward/self.numberOfVisit)

    def rollout(self):
        """
        rollout: run a simulation with randomly choose a child to visit
        return a value of state score

        """
        cState = self.state
        while not cState.isGameOver():
            nbvps = cState.neighbors()
            vpIdx = nbvps[np.random.randint(len(nbvps))]  # rollout policy
            vp = self.util.viewpoints[vpIdx]
            cState = cState.move(vp)
        # print("-- rollout for viewpoint {}, coverage {:.2f} %, score {:.2f}".format(self.viewpoint().id, cState.coverage*100 , cState.score()))
        return cState.score()

    def expand(self):
        """
        expand the child by randomly choose a untried vp to visit
        """
        vpIdx = self.untriedVps.pop(np.random.randint(len(self.untriedVps)))
        vp = self.util.viewpoints[vpIdx]
        nextState = self.state.move(vp)
        child = MCTSNode(self.util, nextState, parent=self)
        self.children.append(child)
        return child

    def backpropagate(self,reward):
        self.numberOfVisit += 1.
        self.totalReward += reward
        if self.maxReward < reward:
            self.maxReward = reward
        if self.parent:
            self.parent.backpropagate(reward)


class MonteCarloTreeSearch(object):
    def __init__(self, node, cparam, decay, targetCoverage):
        self.root = node
        self.epsilon = 1.0
        self.cparam = cparam
        self.decay = decay
        self.targetCoverage = targetCoverage

    def decayEpsilon(self, finalEpsilon=0.1):
        """
        a epsilon with a decay rate until a final eplison is reached
        """
        self.epsilon *= self.decay
        if self.epsilon <= finalEpsilon:
            self.epsilon = finalEpsilon
        return self.epsilon

    def treePolicy(self):
        """
        Tree policy: build up a a tree
        return a best child if the node is fully expand or a random
        otherwise expand the node with a simulation (rollout, Mento Carlo Method)
        """
        node = self.root
        while not node.isTerminalNode():
            if node.isFullyExpanded():
                node = node.best_child(self.cparam, self.epsilon)
            else:
                return node.expand()
        return node

    def search(self,iteration,fe):
        t0 = time.clock()
        for i in range(iteration):
            self.decayEpsilon(finalEpsilon=fe)
            v = self.treePolicy()
            r = v.rollout()
            v.backpropagate(r)
            vps, coverage = self.test()
            print("iteration {}, epsilon {:.4f}, viewpoints explore {}, coverage {:.2f}%".format(i,self.epsilon,len(vps),coverage*100))
            if coverage >= self.targetCoverage:
                print("desired coverage achieved {:.2f}%".format(self.targetCoverage*100))
                break
        t1 = time.clock()
        print("Monte Carlo Tree Search is done in {:.3f} secs".format(t1-t0))
        return self.root.best_child(self.cparam, 0.0)

    def test(self):
        """
        Test the tree
        """
        viewpoints = []
        coverage = 0.0
        node = self.root
        while not node.isTerminalNode():
            vp = node.viewpoint()
            viewpoints.append(vp)
            coverage = node.state.coverage

            if node.isFullyExpanded():
                node = node.best_child(c=self.cparam,e=0.0)
            else:
                break

        return viewpoints, coverage

############################################################
# main
def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--mapfile', type=str, default=None)
    parser.add_argument('--vpsfile', type=str, default=None)
    parser.add_argument('--trajfile', type=str, default=None)

    parser.add_argument('--cn', type=float, default=0.5) # control parameter for neighbors choice
    parser.add_argument('--sn', type=int, default=50) # simulation count
    parser.add_argument('--ad', type=int, default=8) # action dimenstion, how many neighbors
    parser.add_argument('--tc', type=float, default=0.95) # target coverage
    parser.add_argument('--cp', type=float, default=0.38) # control param q value
    parser.add_argument('--dr', type=float, default=0.998) # decay rate
    parser.add_argument('--fe', type=float, default=0.1) # final epsilon

    return parser.parse_args()

if __name__ == "__main__":
    args = getParameters()
    vps = []
    map = GridMap()
    width, height = 0,0
    if args.load:
        map.loadMap(os.path.join(args.load, args.mapfile))
        width, height = map.width, map.height
        vps = loadViewpoints(os.path.join(args.load, args.vpsfile),map)

    # monte carlo tree search
    startIdx = np.random.randint(len(vps))
    startVp = vps[startIdx]
    util = MCTSUtil(map, vps, actDim=args.ad, cn=args.cn)
    initState = util.initialState(startVp)
    root = MCTSNode(util,initState,parent=None)
    mcts = MonteCarloTreeSearch(root,cparam=args.cp,decay=args.dr,targetCoverage=args.tc)
    mcts.search(iteration=args.sn,fe=args.fe)

    bestvps, coverage = mcts.test()
    print("Monte Carlo Tree Search find {} viewpoints for {:.2f}% coverage.".format(len(bestvps), coverage*100))
    saveViewpoints(os.path.join(args.load, args.trajfile), bestvps)

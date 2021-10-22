"""
Ant Colony Optimization
The ACO algorithm is inspired by the foraging behavior of ants. The behavior of the ants are
controled by two main prameters: alpha, or the pheromone's attractiveness to the ant, and beta,
or the exploration capability of the ant. If alpha is very large, the pheromones left by previous
ants in a certain path will be deemed very attractive, making most of the ants divert its way towards
only one route (exploitation), if beta is large, ants are more indepdent in finding the best path
(exploration).
https://ljvmiranda921.github.io/notebook/2017/01/18/ant-colony-optimization-tsp/

The ACO has also been use to produce near-optimal solutions to the TSP. They have an advantage over
simulated annealing and genetic algorithm approaches of similar problems when the graph may change
dynamically; the ACO can be run continuously and adapt to changes in real time. This is of interest
in network routing and urban transportation systems.
The first ACO algorithm was called the ant system and it was aimed to solve the traveling salesman problem,
in which the goal is to find the shortest round-trip to link a series of cities. The general algorithm
is relatively simple and based on a set of ants, each making one of the possible round-trips along the cities.
Ay each stage, the ant chooses to move from on city to another city acoording to some rules:
1. It must visit each city exactly once;
2. A distant city has less chance of being chosen (the visibility);
3. The more intense the pheromone trail laid out on an edge between two cities, the greater the probability
that edge will be chosen;
4. Having completed its journey, the ant deposits more pheromenes on all edges it traversed, if the journey is
short;
5. After each iteration, trail of pheromones evaporate.
https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms#Example_pseudo-code_and_formula
"""
from map import *
from utils import *
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import pandas as pd

MAXFACTOR = 3
EPSILON = 1e-6

"""
In the ant colony optimization algorithms, an artificial ant is a simple computational agent that
searches for good solutions to a given optimization problem.
In the first step of each iteration, each ant stochastically constructs a solution, i.e. the order
in which the edges in the graph should be followed.
In the second step, the paths found by the different ants are compared. The last step consists of
updating the pheromone levels on each edge.

procedure ACO_MetaHeuristic is
    while not terminated do
        generateSolutions()
        daemonActions()
        pheromoneUpdate()
    repeat
end procedure

"""
class Ant:
    """
    Single Ant
    Create a single ant with its properties
    :param int size: the dimension or length of the ant
    """
    uid = 0
    def __init__(self, size):
        self.uid = self.__class__.uid
        self.__class__.uid += 1

        self.size = size
        self.tourLength = np.inf
        self.tour = np.ones(self.size, dtype=np.int64)*-1
        self.visited = np.zeros(self.size, dtype=np.int64)

    def clone(self):
        """
        Returns a deep copy of the current Ant instance with a new UID
        """
        ant = Ant(len(self.tour))
        ant.tourLength = self.tourLength
        ant.tour = self.tour.copy()
        ant.visited = self.visited.copy()
        return ant



class ACO:
    """
    The Ant Colony Optimization metaheuristic
    :param cities: cities' coordinates
    :param ants: number of ants in the colony
    :param maxIter: maximum number of iterations of the algorithm
    :param alpha: the pheromone trail influence
    :param beta: the heuristic information influence
    :param rho: the pheromone evaporation parameter
    """
    def __init__(self, cities, ants = -1, maxIter = 500, alpha = 1.0, beta = 2.0, rho = 0.5):
        self.cities = cities
        self.ants = ants
        self.maxIter = maxIter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.start = None
        self.initialize()

    def initialize(self):
        # initialize the problem
        self.n = len(self.cities)
        self.distMatrix = self.computeDistances(self.cities)
        self.nnList = self.computeNearestNeighbor(self.n,self.distMatrix)
        self.CNN = self.computeNNTourLength(self.n,self.distMatrix) # initial tour
        # initial the colony
        self.eta = 1.0 / (self.distMatrix + 0.1) # add a small constant to avoid zero
        self.iter = 0
        self.bestSoFarAnt = Ant(self.n)
        self.foundBest = 0
        self.restartBestAnt = Ant(self.n)
        self.restartFoundBest = 0
        self.colony = self.createColony(self.ants, self.n)
        self.pheromone = self.resetPheromone(self.ants/self.CNN)
        self.choiceInfo = self.computeChoiceInfo(self.pheromone)

    def computeDistances(self, cities):
        dim = len(cities)
        distMatrix = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                distMatrix[i][j] = distance(cities[i],cities[j])
        # Asign huge values to diagonal in distance matrix, making the distance
        # from a point to itself greater than the maximum
        rowMax = np.amax(distMatrix, axis=1) * MAXFACTOR
        return distMatrix + np.eye(dim) * rowMax

    def computeNearestNeighbor(self, dim, distMatrix):
        """
        Get the nearest-neighbor list of each city
        the nearest-neighbor list, nnList[i][r] gives the identifier(index)
        of the r-th nearest city to city i (i.e. nnList[i][r] = j)
        """
        nn = []
        for row in self.distMatrix:
            d = row.tolist()
            indices = np.lexsort((range(dim), d))
            nn.append(indices)
        return np.array(nn)

    def computeNNTourLength(self,dim, distMatrix):
        """
        A TSP tour generated by the nearest-neighbor heuristic
        """
        tour = np.ones(dim, dtype=np.int64)*-1
        visited = np.zeros(dim, dtype=np.int64)

        step = 0
        r = np.random.randint(0,dim) # initial to random city
        tour[step] = r
        visited[r] = 1
        while (step < dim-1):
            step+=1
            current = tour[step-1]
            next = dim-1
            minDist = np.inf
            for city in range(dim):
                if not visited[city]:
                    dist = distMatrix[current][city]
                    if dist < minDist:
                        next = city
                        minDist = dist
            tour[step] = next
            visited[next] = 1
        # return the tour length
        return self.computeTourLength(tour)

    def computeTourLength(self,tour):
        return sum(distance(self.cities[tour[i]],self.cities[tour[i-1]]) for i in range(len(tour)))

    def createColony(self, numOfAnts, size):
        """Create a colony of ants according to the number of ants specified,"""
        colony = []
        for i in range(numOfAnts):
            colony.append(Ant(size))
        return np.array(colony)

    def resetPheromone(self, level=0.1):
        """Reset the pheromone to a default level"""
        pheromone = np.ones((self.n, self.n), dtype=np.float) * level
        return pheromone

    def computeChoiceInfo(self, pheromone):
        """
        Compute the choice information matrix using the pheromone and heuristic information.
        """
        return pheromone**self.alpha*self.eta**self.beta

    def run(self):
        progress = []
        t0 = time.clock()
        print("*** Running Ant Colony Optimization ***")
        while self.iter < self.maxIter:
            self.generateSolutions()
            self.updateStatistics()
            self.updatePheromone()
            self.iter += 1

            # console output
            lenValues = np.array([ant.tourLength for ant in self.colony])
            progress.append(np.amin(lenValues))
            stats = [self.iter,np.amax(lenValues),np.amin(lenValues),np.mean(lenValues),np.std(lenValues)]
            print("{0}\t{1}\t{2}\t{3}\t{4}".format(stats[0], stats[1], stats[2], stats[3], stats[4]))
        t1 = time.clock()
        bestTour = [self.cities[i] for i in self.bestSoFarAnt.tour]
        print("{} city tour with length {:.2f} in {:.3f} secs".format(len(self.cities), tour_length(bestTour), t1-t0))
        bestTour = alter_tour(bestTour)
        return progress, bestTour

    def generateSolutions(self):
        """Construct valid solutions for the TSP."""
        step = 0
        # 1. Clear ants memory
        for ant in self.colony:
            for i in range(len(ant.visited)):
                ant.visited[i] = 0
        # 2. Assign an initial random city to each ant
        for ant in self.colony:
            r = np.random.randint(0, self.n)
            ant.tour[step] = r
            ant.visited[r] = 1
        # 3. Each ant constructs a complete tour
        while step < self.n-1:
            step += 1
            for k in range(self.ants):
                self.decisionRule(k,step)
        # 4. Move to initial city and compute each ant's tour length
        for ant in self.colony:
            ant.tourLength = self.computeTourLength(ant.tour)

    def decisionRule(self, k, i):
        """
        The ants apply the Ant System (AS) action choice rule eq.3.2
        :param int k: ant identifier
        :param int i: counter for construction step
        """
        c = self.colony[k].tour[i-1] # current city
        # create a roulette wheel, like in evolutionary computation
        # sum the probabilities of the cities not yet visited
        sumProp = 0.0
        selectProb = np.zeros(self.n, dtype=np.float)
        for j in range(self.n):
            if self.colony[k].visited[j]:
                selectProb[j] = 0.0 # if city has been visited, its probability is zero
            else:
                # assign a slice to the roulette wheel, proportional to the weight of the associated choice
                selectProb[j] = self.choiceInfo[c][j]
                sumProp += selectProb[j]

        # Spin the roulette wheel
        # Random number from the interval [0, sumProb], corresponding to a uniform distribution
        r = sumProp*np.random.random_sample()
        j = 0
        p = selectProb[j]
        while p < r:
            j += 1
            p += selectProb[j]

        self.colony[k].tour[i] = j
        self.colony[k].visited[j] = 1

    def updateStatistics(self):
        """
        Manage some statistical information about the trial, especially
        if a new best solution (best-so-far or restart-best) if found and
        adjust some parametyers if a new best solution is found
        """
        iterBestAnt = self.findBest()
        # Update best so far ant
        diff = self.bestSoFarAnt.tourLength - iterBestAnt.tourLength
        if diff > EPSILON:
            self.bestSoFarAnt = iterBestAnt.clone()
            self.restartBestAnt = iterBestAnt.clone()
            self.foundBest = self.iter
            self.restartFoundBest = self.iter

        # Update restart best ant
        diff = self.restartBestAnt.tourLength - iterBestAnt.tourLength
        if diff > EPSILON:
            self.restartBestAnt = iterBestAnt.clone()
            self.restartFoundBest = self.iter

    def updatePheromone(self):
        """
        Pheromone trail update
        Pheromone trail are evaporated and pheromones are deposited according to
        the rules defined by the various ACO agorithms
        """
        # decreases the values of the pheromone trails on all the arcs by a constant
        # factor rho. This uses matrix operation
        self.pheromone = self.pheromone*(1.0-self.rho)
        for ant in self.colony:
            # adds pheromone to the arcs belonging to the tours constructed by ant
            delta = 1.0 / ant.tourLength
            for i in range(self.n-1):
                j = ant.tour[i]
                k = ant.tour[i+1]
                self.pheromone[j][k] = self.pheromone[j][k] + delta
                self.pheromone[k][j] = self.pheromone[j][k]
        # compute the choice information matrix using the pheromone and heuristic information
        self.choiceInfo = self.computeChoiceInfo(self.pheromone)

    def findBest(self):
        """
        FInd the best ant object from the colony in the current iteration
        """
        best = self.colony[0]
        for ant in self.colony:
            if ant.tourLength < best.tourLength:
                best = ant.clone()
        return best


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default="Random")
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--ants', type=int, default=10)
    parser.add_argument('--maxIter', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--rho', type=float, default=0.5)
    return parser.parse_args()

# main loop
if __name__ == '__main__':
    #cities = USA_landmarks_map()
    args = getArgs()
    cities = Random_cities_map(args.size)
    if args.map == "USA":
        cities = USA_landmarks_map()

    print(list(cities))
    tspACO = ACO(cities=list(cities), ants=args.ants, maxIter=args.maxIter, alpha=args.alpha, beta=args.beta, rho=args.rho)
    progress, bestTour = tspACO.run()

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Iteration')
    plt.show()
    plot_tour(bestTour)
    plt.show()

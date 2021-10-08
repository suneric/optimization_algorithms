"""
Genetic Algorithm
A GA is a search heuristic that is inspired by Charles Darwin's theory of natural evolution.
This algorithm reflects the process of natural selection where the fittest individuals are
selected for reproduction in order to produce offspring of the next generation.
The process of natural selection starts with the selection of fittes individuals from a population.
They produce offspring which inherrit the characteristics of the parents and will be added to the
next generation. If parents have better fitness, their offspring will be better than parents and have
a better chance at surviving. This process keeps on iterating and at the end, a generation with the
fittest individuals will be found.
Notation in context of TSP:
Gene: a city (represented a (x,y) coordinates)
Individual (aka "chromosome"): a single route satisfying the condition above
Population: a collection of possible routes (i.e. collection of individuals)
Parents: two routes that are combined to create a new route
Mating pool: a collection of parents that are used to create out next population
Fitness: a function that tells us how good each route is (how short the distance is)
Mutation: a way to introduce variation in our population by randomly swapping two cities in a route
Elitism: a way to carry the best individuals into the next generation.
"""
from map import *
from utils import *
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import pandas as pd

"""
Create the population
"""
def initialPopulation(popSize, cityList, seed=42):
    random.seed(seed*popSize)
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def createRoute(cityList):
    # random.sample(population, k): return k length list of unique elements chosen
    # from the population sequence.
    route = random.sample(cityList, len(cityList))
    return route


"""
Determine fitness
"""
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0.0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            self.distance = tour_length(self.route)
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1/float(self.routeDistance())
        return self.fitness

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

"""
Select the mating pool
The most common approach are either fitness propotionate selection (aka roulette wheel selection)
or tournamenet selection:
Fitness proportionate selection: The fitness of each individual relative to the population is used to'
assign a probability of selection. Think of this as the fitness-weighted probability of being selected.
Tournament selection: A set number of individuals are randomly selected from the population and the one
with the highest fitness in the group is chosen as the first parant. This is repeated to chose the
second parent.
Elitism: the best performing individuals from the population will automatically carry over to the next
generation, ensuring that the most successful individuals persist.
https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
"""
def selection(popRanked, eliteSize):
    selectionResults = []
    # select the best performed individuals
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    # fitness proportionate selection
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum() # cumulative sum of fitness
    df['cum_perc'] = df.cum_sum/df.Fitness.sum() # percentage of each individual
    for i in range(0, len(popRanked)-eliteSize):
        selectionResults.append(roulette_selection(popRanked,df))

    return selectionResults

def roulette_selection(popRanked, df):
    pick = random.random()
    for i in range(0, len(popRanked)):
        if pick < df.iat[i,3]:
            return(popRanked[i][0])

def matingPool(population, selectionResults):
    matingPool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingPool.append(population[index])
    return matingPool

"""
Breed (crossover)
Create the next generation in a process called crossover (aka breeding)
The TSP is unique in that we need to include all locations exactily one time. To abide by this rule,
we can use a special breeding function called ordered crossover. we randomly select a subset of the
first parent string and then fill the remainder of the route with the genes from the second parent in
the order in which they appear, without duplicating any genes in the selected subset from the first parent
"""
def crossover(a, b):
    child = []
    childA, childB = [],[]
    geneA = int(random.random()*len(a))
    geneB = int(random.random()*len(a))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childA.append(a[i])

    childB = [item for item in b if item not in childA]
    child = childA + childB
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    for i in range(0, eliteSize):
        children.append(matingpool[i])
    pool = random.sample(matingpool, len(matingpool))
    for i in range(0, len(matingpool)-eliteSize):
        children.append(crossover(pool[i], pool[i+1]))
    return children


"""
Mutate
Mutation serves an important function in GA, as it helps to avoid local convergences by introducing
novel routes that will allow us to explore other parts of the solution space.
TSP has a special cosideration when it comes to mutation. We can not drop cities. Instead, we use
swap mutation with specified low probability.
"""
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random()*len(individual))
            temp1 = individual[swapped]
            temp2 = individual[swapWith]
            individual[swapWith] = temp1
            individual[swapped] = temp2
    return individual

def mutatePopulation(population, mutationRate):
    new_generation = []
    for individual in range(0,len(population)):
        mutatedIndividual = mutate(population[individual], mutationRate)
        new_generation.append(mutatedIndividual)
    return new_generation


"""
Repeat
"""
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGen = mutatePopulation(children, mutationRate)
    return nextGen


def geneticAlgorithm(cities, popSize, eliteSize, mutationRate, generations):
    t0 = time.clock()

    progress = []
    pop = initialPopulation(popSize, cities)
    rank = 1/rankRoutes(pop)[0][1]
    progress.append(rank)
    print("Initial distance:"+str(rank))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        rank = 1/rankRoutes(pop)[0][1]
        progress.append(rank)
        print("Generation " + str(i) + " Distance: " + str(rank))

    t1 = time.clock()
    print("{} city tour with length {:.1f} in {:.3f} secs".format(len(cities), 1/rankRoutes(pop)[0][1], t1-t0))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute, progress


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default="Random")
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--popSize', type=int, default=100)
    parser.add_argument('--eliteSize', type=int, default=20)
    parser.add_argument('--mutationRate', type=float, default=0.01)
    parser.add_argument('--generations', type=int, default=500)
    return parser.parse_args()

# main loop
if __name__ == '__main__':
    args = getArgs()
    cities = Random_cities_map(args.size)
    if args.map == "USA":
        cities = USA_landmarks_map()
    bestTour, progress = geneticAlgorithm(
                            cities=cities,
                            popSize=args.popSize,
                            eliteSize=args.eliteSize,
                            mutationRate=args.mutationRate,
                            generations=args.generations)
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    plot_tour(bestTour)
    plt.show()

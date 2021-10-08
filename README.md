# Optimization Algorithms


## Traveling Salesman Problem (TSP)
The [Traveling Salesman Problem (TSP)](http://www.math.uwaterloo.ca/tsp/) is one of the most intensive studied problems in computational mathematics.
**Given a set of cities and the distances between each pair of cities, what is the shortest possible tour that visits each city exactly once, and returns to the starting city?**

### Solutions
- Optimal
  - Brute-Force Approach
  - Dynamic Programming
- Approximate Optimal
  - Nearest Neighbor Algorithm
  - Greedy Algorithm
  - Minimum Spanning Tree
  - Divide and Conquer
- Population-Based Optimization
  - Genetic (Evolutionary) Algorithm
  - Ant Colony Optimization
  - Particle Swarming Optimization

The difference between traditional algorithms and EAs is that EAs are not static but dynamic as they can evolve over time. Evolutionary algorithms have three main characteristics:
1. **Population-Based**: EAs are to optimize a process in which current solutions are bad to generate new better solutions. The set of current solutions from which new solutions are to be generated is called the population.
2. **Fitness-Oriented**: If there are some several solutions, how to say that one solution is better than another? There is a fitness value associated with each individual solution calculated from a fitness function. Such fitness value reflects how good the solution is.
3. **Variation-Driven**: If there is no acceptable solution in the current population according to the fitness function calculated from each individual, we should make something to generate new better solutions. As a result, individual solutions will undergo a number of variations to generate new solutions.

### References
1. TSP: http://www.math.uwaterloo.ca/tsp/
2. TSP Algorithms: http://www.exatas.ufpr.br/portal/docs_degraf/paulo/TravellingSalesmanProblem.pdf
3. TSP Algorithms: https://nbviewer.jupyter.org/url/norvig.com/ipython/TSP.ipynb
4. GA: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
5. ACO: https://github.com/ecoslacker/yaaco


## Set Covering Problem (SCP)
The set covering problem is a significant NP-hard problem (which means the time required to find the answer grows dramatically as the problem size increases). Given a collection of elements, the set covering problem aims to find the minimum number of sets that cover (incorporate) all of these elements.
In the set covering problem, two sets are given: a set U of elements and a set S of subsets of the set U. Each subset in S is associated with a predetermined cost, and the union of all the subsets covers the set U. This combinatorial problem then concerns finding the optimal number of subsets whose union covers the universal set while minimizing the total cost.

### Solutions
- Greedy Minimum Set Cover
- Linear Polynomial Rounding

### References
1. https://optimization.cbe.cornell.edu/index.php?title=Set_covering_problem


## Discrete Coverage Path Planning Problem (CPP)

### Solutions
- Optimization Approach: SCP + TSP
- Reinforcement Approach:
  - Monte Carlo Tree Search
  - q-Learning
  - PPO
  - DDPG

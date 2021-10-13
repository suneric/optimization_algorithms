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

# base Node
class MCTSNode(object):
    def __init__(self,util,state,parent=None):
        self.util = util
        self.parent = parent
        self.children = []
        self.state = state
        self.neighbors = self.state.unvisited_neighbors()
        self.untriedVps = [self.neighbors[i] for i in range(len(self.neighbors))]

    def vp_index(self):
        return self.state.vp()

    def random_child(self):
        return self.children[np.random.randint(len(self.children))]

    def child_nodes(self):
        return self.children
    def parent(self):
        return self.parent

    def is_terminal_node(self):
        return self.state.coverage() == 1.0 or len(self.neighbors) == 0

    def is_fully_expanded(self):
        return len(self.untriedVps) == 0

    def rollout_policy(self, possibleMoves):
        return possibleMoves[np.random.randint(len(possibleMoves))]

# Node of traveling reward
class MCTSNodeReward(MCTSNode):
    def __init__(self, util, state, parent=None):
        MCTSNode.__init__(self,util,state,parent)
        self.totalReward = 0.
        self.numberOfVisit = 0.
        self.maxReward = 0.

    def best_child(self, c_param=0.618):
        weights = [child.q(c_param) for child in self.children]
        return self.children[np.argmax(weights)]

    def q(self,alpha):
        return (1-alpha)*self.maxReward + alpha*(self.totalReward/self.numberOfVisit)

    def rollout(self, target_coverage):
        cState = self.state
        while cState.coverage() < target_coverage:
            neighbors = cState.unvisited_neighbors()
            if len(neighbors) == 0:
                neighbors = cState.neighbors()
            nextVp = self.rollout_policy(neighbors)
            cState = cState.move(nextVp)
        return 100*(cState.coverage()/cState.travel_distance())

    def expand(self):
        vpIdx = self.untriedVps.pop(np.random.randint(len(self.untriedVps)))
        nextState = self.state.move(vpIdx)
        child = MCTSNodeReward(self.util, nextState, parent=self)
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
    def __init__(self, node):
        self.root = node
        self.epsilon = 1.0

    def decay_epsilon(self, decay_rate, final_eps=0.1):
        self.epsilon *= decay_rate
        if self.epsilon <= final_eps:
            self.epsilon = final_eps
        return self.epsilon

    def tree_policy(self,cparam,epsilon):
        node = self.root
        while not node.is_terminal_node():
            if node.is_fully_expanded():
                if np.random.rand() > epsilon:
                    node = node.best_child(cparam)
                else:
                    node = node.random_child()
            else:
                return node.expand()
        return node

    def search(self,simulation_number,target,cparam,dr,fe,env):
        for i in range(simulation_number):
            epsilon = self.decay_epsilon(dr,fe)
            v = self.tree_policy(cparam,epsilon)
            r = v.rollout(target)
            v.backpropagate(r)
            if i%10 == 0:
                s = int(i/10)
                vps, coverage = self.test(cparam,env)
                print("test: iteration(x10)", s, "epsilon", self.epsilon, "visited", vps, "coverage", coverage)
                if coverage >= target:
                    print("desired coverage achieved", target)
                    break
        return self.root.best_child()

    def test(self, cparam, env, saved=None):
        node = self.root
        state = env.reset()
        step = 1
        while not node.is_terminal_node():
            done, reward, coverage, state = env.play(node.vp_index())
            if node.is_fully_expanded():
                node = node.best_child(cparam)
                step += 1
            else:
                break
        # save trajectory
        if saved != None:
            env.save(saved)
            print("save trajectory to ", saved)

        return step, coverage

############################################################
# main
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sn', type=int, default=50) # simulation count
    parser.add_argument('--ad', type=int, default=8) # action dimenstion, how many neighbors
    parser.add_argument('--tc', type=float, default=0.95) # target coverage
    parser.add_argument('--cp', type=float, default=0.38) # control param
    parser.add_argument('--dr', type=float, default=0.998) # decay rate
    parser.add_argument('--fe', type=float, default=0.1) # final epsilon
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    simulation_count = args.sn
    action_dim = args.ad
    target_coverage = args.tc
    cparam = args.cp
    dr = args.dr
    fe = args.fe

    state = env.reset()
    root = MCTSNodeReward(env.utility(), state, parent=None)
    mcts = MonteCarloTreeSearch(root)
    mcts.search(simulation_count,target_coverage,cparam,dr,fe,env)
    mcts.test(cparam,env,os.path.join(output_dir,"viewpoints.txt"))

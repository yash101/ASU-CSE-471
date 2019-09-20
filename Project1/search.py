# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearchRecursive(problem, parent, expand, direction, pathstk, pathvisited):
    if problem.isGoalState(expand):
        pathstk.push((expand, direction))
        return True
    
    if expand in pathvisited:
        return False
    
    # get the next possible states
    nextStates = problem.getSuccessors(expand)
    pathstk.push((expand, direction))
    pathvisited[expand] = True

    for state in nextStates:
        if depthFirstSearchRecursive(problem, expand, state[0], state[1], pathstk, pathvisited):
            return True
    
    pathstk.pop()
    del pathvisited[expand]
    return False

def depthFirstSearch(problem): # recursive algorithm that is more likely to work
    sys.setrecursionlimit(1500)
    pathstk = util.Stack()
    visited = {}
    depthFirstSearchRecursive(problem, None, problem.getStartState(), None, pathstk, visited)
    path = []
    while not pathstk.isEmpty():
        path.append(pathstk.pop()[1])

    print(path)
    
    return path[::-1][1:]

def breadthFirstSearch(problem):
    start, goal = (problem.getStartState(), None), None
    q = util.Queue()
    visited, parents = {}, {}

    q.push(start)
    parents[start[0]] = None
    visited[start[0]] = True

    while not q.isEmpty():
        cur = q.pop()
        c_coords, c_dir = cur[0], cur[1]

        if problem.isGoalState(c_coords):
            goal = (c_coords, c_dir)
            break

        neighbors = problem.getSuccessors(c_coords)

        for neighbor in neighbors:
            n_coords, n_dir = neighbor[0], neighbor[1]

            if n_coords in visited:
                continue

            visited[n_coords] = True
            parents[n_coords] = (c_coords, n_dir)

            q.push((n_coords, n_dir))

    pathrev = [ goal ]
    while parents[ pathrev[-1][0] ] is not None:
        pathrev.append(parents[ pathrev[-1][0] ])
    
    path = [step[1] for step in pathrev[::-1]]
    return path[:-1]


def uniformCostSearch(problem):
    start, goal = (problem.getStartState(), []), None
    queue = util.PriorityQueue()
    visited = {}
    c_path = None

    queue.push((start[0], [], 0), 0)
    while not queue.isEmpty():
        cur = queue.pop()
        c_coords, c_path, c_cost = cur[0], cur[1], cur[2]

        if problem.isGoalState(c_coords):
            goal = (c_coords, c_path)
            break

        if c_coords not in visited:
            visited[c_coords] = True
            
            neighbors = problem.getSuccessors(c_coords)
            for neighbor in neighbors:
                n_coords, n_path, n_cost = neighbor[0], neighbor[1], neighbor[2]
                path = c_path + [n_path]
                cost = c_cost + n_cost
                queue.push((n_coords, path, cost), cost)
    
    return c_path


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    start, goal = (problem.getStartState(), []), None
    queue = util.PriorityQueue()
    visited = {}
    c_path = None

    queue.push((start[0], [], 0), 0)
    while not queue.isEmpty():
        cur = queue.pop()
        c_coords, c_path, c_cost = cur[0], cur[1], cur[2]

        if problem.isGoalState(c_coords):
            goal = (c_coords, c_path)
            break

        if c_coords not in visited:
            visited[c_coords] = True
            
            neighbors = problem.getSuccessors(c_coords)
            for neighbor in neighbors:
                n_coords, n_path, n_cost = neighbor[0], neighbor[1], neighbor[2]
                path = c_path + [n_path]
                cost = c_cost + n_cost
                queue.push((n_coords, path, cost),
                    cost + heuristic(n_coords, problem))
    
    return c_path

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

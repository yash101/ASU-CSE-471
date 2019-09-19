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

def depthFirstSearch(problem):

    stk = util.Stack()
    stk.push(problem.getStartState())
    visited = { problem.getStartState(): True }
    parents = { problem.getStartState(): None }
    successors = {}
    goal = None

    while not stk.isEmpty():
        cur = stk.pop()
        
        if problem.isGoalState(cur):
            goal = cur
            break

        nxt = problem.getSuccessors(cur)
        successors[cur] = nxt

        for move in nxt:
            if move[0] in visited:
                continue

            parents[move[0]] = cur
            
            visited[move[0]] = True
            stk.push(move[0])

    path = [ goal ]

    while path[-1] is not None:
        path.append(parents[path[-1]])

    imnotlost = []

    actualpath = path[::-1][1:]
    for i in range(len(actualpath) - 1):
        step = actualpath[i]
        nxt = actualpath[i + 1]

        poss = successors[step]
        for move in poss:
            if move[0] == nxt:
                imnotlost.append(move[1])

#    print("Im not actully lost: ", imnotlost)

    return imnotlost

#     s = util.Stack() # contains tuples of positions
#     parents = {}
#     visited = {}
#     parents[problem.getStartState()] = None # [ ( , ) ]
#     goal = None

#     s.push(problem.getStartState())

#     while not s.isEmpty():
#         print(s)
#         element = s.pop()

#         if problem.isGoalState(element):
#             goal = element
#             break

#         successors = problem.getSuccessors(element)
        
#         for successor in successors:
#             # If this successor doesn't have a parent, we want to set this as our parent
#             if not successor[0] in parents:

#                 s.push(successor[0])

#     start = problem.getStartState()

#     print(parents)

#     # build the path now
#     path = [goal]
#     print(goal)
#     while parents[path[-1]] is not None:
#         path.append(parents[path[-1]])
# #        print(path)
    
#    print(path)


    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

# multiAgents.py
# --------------
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

from __future__ import division

from util import manhattanDistance
from game import Directions
import random, util
import math
import sys

from game import Agent

def distance(a, b, fn):
    dx, dy = b[0] - a[0], b[1] - a[1]
    if fn is None or fn is 'euclidian':
        return math.sqrt((dx * dx) + (dy * dy))
    else:
        return util.manhattanDistance(a, b)

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

#        scaredTime = 0
#        for time in newScaredTimes:
#            scaredTime += time

#        return (1. / (1 + len(newFood.asList()))) + scaredTime

# time to go more advanced because the simple stuff isn't working

        # Tuning params for this code:
        # proximalDistance, wScore, wFFD, wTGC, wNNG, wST


        # reciprocal of the furthest byte of food

        furthestFood, furthestFoodDistance = None, 0
        foods = newFood.asList()

        for food in foods:
            # euclidian distance to food
            fDistance = distance(food, newPos, 'euclidian')
            if furthestFood is None or furthestFoodDistance > fDistance:
                furthestFood = food
                furthestFoodDistance = fDistance
    
        totalGhostDistance = 0
        numberNearbyGhosts = 0

        # param for distance to consider a ghost as nearby:
        proximalDistance = 2

        for ghost in successorGameState.getGhostPositions():
            dist = distance(ghost, newPos, 'euclidian')
            totalGhostDistance += dist

            if dist <= proximalDistance:
                numberNearbyGhosts += 1
        
        scaredTime = 0
        for time in newScaredTimes:
            scaredTime += time

        # tuning params
        wScore, wFFD, wTGD, wNNG, wST = 1.1, 1.3, -1.1, -1., 1.0

        return sum([
            wScore * successorGameState.getScore(),
            wFFD / (1. + furthestFoodDistance),
            wTGD / (1. + totalGhostDistance),
            (wNNG * numberNearbyGhosts) * (wST * (0 - scaredTime))
        ])

#        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, agent, depth, gameState):
        if depth is self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(agent)
        if agent is 0:
            mm = [self.minimax(1, depth, gameState.generateSuccessor(agent, state)) for state in legalActions]
            return max(mm)

        else:
            nextAgent = agent + 1

            if gameState.getNumAgents() is nextAgent:
                nextAgent = 0

            if nextAgent is 0:
                depth += 1
            
            mm = [self.minimax(nextAgent, depth, gameState.generateSuccessor(agent, state)) for state in legalActions]

            return min(mm)


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        maximum, action = None, None
        
        for state in gameState.getLegalActions(0):

            mm = self.minimax(1, 0, gameState.generateSuccessor(0, state))
            if maximum is None or mm > maximum:
                maximum = mm
                action = state

        return action

def NoneMin(a, b):
    if a is None:
        return b
    elif b is None:
        return a
    elif a < b:
        return a
    else:
        return b

def NoneMax(a, b):
    if a is None:
        return b
    elif b is None:
        return a
    elif a > b:
        return a
    else:
        return b

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def maximize(self, agent, depth, state, alpha, beta):
        val = float('-inf')

        legalActions = state.getLegalActions(agent)
        for action in legalActions:
            val = NoneMax(
                val,
                self.alphabetaprune(
                    1,
                    depth,
                    state.generateSuccessor(agent, action),
                    alpha,
                    beta
                )
            )

            if val > beta:
                return val

            alpha = NoneMax(alpha, val)

        return val

    def minimize(self, agent, depth, state, alpha, beta):
        val = float('inf')
        next = (agent + 1) % state.getNumAgents()

        if next is 0:
            depth += 1

        legalActions = state.getLegalActions(agent)
        for action in legalActions:
            val = NoneMin(
                val,
                self.alphabetaprune(
                    next,
                    depth,
                    state.generateSuccessor(agent, action),
                    alpha,
                    beta
                )
            )

            if val < alpha:
                return val

            beta = NoneMin(beta, val)

        return val

    def alphabetaprune(self, agent, depth, state, alpha, beta):

        # base case - check for terminal node
        if state.isLose() or state.isWin() or depth is self.depth:
            return self.evaluationFunction(state)

        # check if we've got a pacman - maximize the pacman
        if agent is 0:
            return self.maximize(
                agent,
                depth,
                state,
                alpha,
                beta
            )

        # check if we are a ghost - minimize the ghost
        else:
            return self.minimize(
                agent,
                depth,
                state,
                alpha,
                beta
            )

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        utility, action, alpha, beta = float('-inf'), None, float('-inf'), float('inf')

        legalActions = gameState.getLegalActions(0)
        
        for act in legalActions:
            val = self.alphabetaprune(
                1,
                0,
                gameState.generateSuccessor(0, act),
                alpha,
                beta
            )

            if val > utility:
                utility = val
                action = act

            if utility > beta:
                return utility

            alpha = NoneMax(alpha, utility)

        return action



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectminimax(self, state, agent, depth):
        # check base case - is the game over or did the search reach its depth limit?
        if depth is self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # if agent is 0:
        #     return NoneMax([
        #         self.expectimax(
        #             state.generateSuccessor(agent, st)
        #             1,
        #             depth
        #         ) for st in state.getLegalActions(agent)
        #     ])        

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # calculate the distance to the nearest and furthest food
    furthestFood, furthestFoodDistance = None, 0
    nearestFood, nearestFoodDistance = None, 0
    foods = newFood.asList()

    for food in foods:
        # euclidian distance to food
        fDistance = distance(food, newPos, 'euclidian')
        if furthestFood is None or fDistance > furthestFoodDistance:
            furthestFood = food
            furthestFoodDistance = fDistance
        if nearestFood is None or fDistance < nearestFoodDistance:
            nearestFood = food
            nearestFoodDistance = fDistance

    totalGhostDistance = 0
    numberNearbyGhosts = 0

    
    scaredTime = 0
    for time in newScaredTimes:
        scaredTime += time

    # If ghosts are scared, we wanna hunt them :P
    # If there are any ghosts within pacman's range while they are scared, attacc
    # param for distance to consider a ghost as nearby:
    proximalDistance = scaredTime + 1

    for ghost in successorGameState.getGhostPositions():
        dist = distance(ghost, newPos, 'euclidian')
        totalGhostDistance += dist

        if dist <= proximalDistance:
            numberNearbyGhosts += 1

    # tuning params
    wScore, wFFD, wTGD, wNNG, wST = 1.1, 1.3, -1.1, -1., 1.0

    return sum([
        wScore * successorGameState.getScore(),
        wFFD / (1. + furthestFoodDistance),
        wTGD / (1. + totalGhostDistance),
        (wNNG * numberNearbyGhosts) * (wST * (0 - scaredTime))
    ])

# Abbreviation
better = betterEvaluationFunction

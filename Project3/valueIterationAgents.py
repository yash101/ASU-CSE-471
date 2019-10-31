# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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

# Sources used:
# https://en.wikipedia.org/wiki/Markov_decision_process
# Class textbook
# Class slides

import mdp, util, math

from learningAgents import ValueEstimationAgent
import collections

def noneMax(vals, defaultValue = None):
    max = None
    for val in vals:
        if val is not None and max is not None and val > max:
            max = val
    if max is None:
        max = defaultValue
    
    return max

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def valueIterationStep(self, iteration):
        # Get all the states in the MDP to iterate over
        S = self.mdp.getStates()
        newValues = self.values.copy()

        for s in S:

            # get the possible actions and their respective states to calculate the value
            # maximize over A
            maxValue = None
            A = self.mdp.getPossibleActions(s)
            for a in A:

                # (nextState, prob)[]
                Sprime = self.mdp.getTransitionStatesAndProbs(s, a)
                for (sprime, Pa) in Sprime:
                    Ra = self.mdp.getReward(s, a, sprime)
                    gamma = (math.pow(self.discount, iteration))
                    Vi = self.values[sprime]
                    val = (Pa * Ra) + (gamma * Vi)
                    maxValue = noneMax([maxValue, val])

            newValues[s] = maxValue
        
        self.values = newValues

    def runValueIteration(self):
        """
        V(i+1, s) = max over a { sum over s' [ Pa(s, s')(Ra(s, s') + gamma Vi(s')) ] }
        """
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            self.valueIterationStep(i)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q(s, a) = sum of sprime ... [the weighted average]
        Q = 0

        # get next possible states and their probabilities
        Sprime = self.mdp.getTransitionStatesAndProbs(state, action)
        for (sprime, Pa) in Sprime:
            if self.mdp.isTerminal(sprime):
                continue

            Ra = self.mdp.getReward(state, action, sprime)
            discount = self.discount
            Vi = self.values[sprime]

            # debug cus of crash
#            print(sprime, Pa, Ra, discount, Vi)

            Q += Pa * (Ra + (discount * Vi))
        
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actionTaken = None

        if self.mdp.isTerminal(state): # If the current state is terminal, no actions necessary :D
            return None
        
        A = self.mdp.getPossibleActions(state)
        maxSum = None

        for a in A:
            avg = self.computeQValueFromValues(state, a)

            if maxSum is None or avg > maxSum or (maxSum is 0.0 and actionTaken is None):
                actionTaken = a
                maxSum = avg
        
        return actionTaken

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"


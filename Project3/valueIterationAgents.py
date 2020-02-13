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
        if val is not None:
            if max is None or val > max:
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

    def valueIterationStep(self):
        """
         Solve for V(k+1, s) for each state s
        """
        # Get all the states in the MDP to iterate over
        S = self.mdp.getStates()
        newValues = self.values.copy()

        for state in S:

            if self.mdp.isTerminal(state):
                continue
            
            maxState, A = None, self.mdp.getPossibleActions(state)

            for action in A:

                sumAvg = 0
                Sprime = self.mdp.getTransitionStatesAndProbs(state, action)
                
                for (sprime, Pa) in Sprime:
                    Ra = self.mdp.getReward(state, action, sprime)
                    gamma = self.discount
                    # gamma = (math.pow(self.discount, iteration))
                    Vi = self.values[sprime]
                    
                    val = Pa * (Ra + (gamma * Vi))
                    sumAvg = sum([sumAvg, val])

                maxState = noneMax([maxState, sumAvg])
            newValues[state] = maxState

        self.values = newValues

    def runValueIteration(self):
        """
        V(i+1, s) = max over a { sum over s' [ Pa(s, s')(Ra(s, s') + gamma Vi(s')) ] }
        """
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # zeroing
        if self.iterations is 0:
            S = self.mdp.getStates()
            for s in S:
                self.values[s] = 0
            return self.values

        while self.iterations > 0:
            self.valueIterationStep()
            self.iterations -= 1

        return self.values

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

            Ra = self.mdp.getReward(state, action, sprime)
            discount = self.discount
            Vi = self.values[sprime]

            # debug cus of crash
#            print(sprime, Pa, Ra, discount, Vi)

            Q += (Pa * (Ra + (discount * Vi)))
        
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
        self.current = None

        ValueIterationAgent.__init__(self, mdp, discount, iterations)


    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):

            if self.current is None:
                self.current = 0
            else:
                self.current = (self.current + 1) % len(self.mdp.getStates())
            
            state = self.mdp.getStates()[self.current]

            if self.mdp.isTerminal(state):
                continue

            maxState, A = None, self.mdp.getPossibleActions(state)

            for action in A:

                sumAvg = 0.0
                Sprime = self.mdp.getTransitionStatesAndProbs(state, action)
                
                for (sprime, Pa) in Sprime:
                    Ra = self.mdp.getReward(state, action, sprime)
                    gamma = self.discount
                    # gamma = (math.pow(self.discount, iteration))
                    Vi = self.values[sprime]
                    
                    val = Pa * (Ra + (gamma * Vi))
                    sumAvg = sum([sumAvg, val])

                maxState = noneMax([maxState, sumAvg])
            self.values[state] = maxState


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
        self.discount = discount
        self.iterations = iterations

        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        return "Not implemented :("

        # V(k+1, s) -> maximize over the action, the sum of ...

        # get the predecessors: (state): [((state), action_taken)]
        predecessors = {}
        q = util.PriorityQueue()
        S = self.mdp.getStates()

        # initialize
        for s in S:
            if self.mdp.isTerminal(s):
                continue

            # get next actions
            A = self.mdp.getPossibleActions(s)
            for a in A:
                Sprime = self.mdp.getTransitionStatesAndProbs(s, a)

                for sprime in Sprime:
                    # s -a-> sprime
                    #   insert: sprime: [(s, a)]

                    if sprime not in predecessors:
                        predecessors[sprime] = [(s, a)]
                    else:
                        predecessors[sprime].append([(s, a)])

        
        # run
        ## Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
        for s in S:
            if self.mdp.isTerminal(s):
                continue

            maxDiff = None
            A = self.mdp.getPossibleActions(s)
            for a in A:
               diff = (self.values[s] - self.computeQValueFromValues(s, a))
               maxDiff = noneMax([maxDiff, diff, -diff])
            
            q.push(s, -float(maxDiff))

        for i in range(self.iterations):
            if q.isEmpty():
                break

            s = q.pop()

            if self.mdp.isTerminal(s):
                continue

            maxState, A = None, self.mdp.getPossibleActions(s)
            for action in A:
                sumAvg = 0.0
                sprime = self.mdp.getTransitionStatesAndProbs(s, action)

                for (sprime, Pa) in Sprime:
                    Ra = self.mdp.getReward(s, action, sprime)
                    gamma = self.discount
                    Vi = self.values[sprime]

                    val = Pa * (Ra + (gamma * Vi))
                    sumAvg = sum([sumAvg, val])
                
                maxState = noneMax([maxState, sumAvg])
            self.values[s] = maxState

            pred = predecessors[s]

            for predecessor in pred:
                PA = self.mdp.getPossibleActions(predecessor)
#                for Pa in PA:
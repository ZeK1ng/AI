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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        k = self.iterations
        for Ki in range(k):
            vector = self.values.copy()
            for st in self.mdp.getStates():
                if self.mdp.isTerminal(st):
                    continue
                best = None
                for action in self.mdp.getPossibleActions(st):
                    if best is None:
                        best = self.getQValue(st,action)
                    else:
                        best = max(best,self.getQValue(st,action))
                vector[st] = best
            self.values=vector 

                    



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
        res = 0
        # print(self.mdp.getTransitionStatesAndProbs(state,action))
        statesAndProbs = self.mdp.getTransitionStatesAndProbs(state,action)
        for next_state,probability in statesAndProbs:
            reward = self.mdp.getReward(state,action,next_state)
            dscFactor= self.discount
            factorTimesval = self.values[next_state]*dscFactor
            reward+=factorTimesval
            res+=reward*probability
        return res

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        p = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            p[action] = self.computeQValueFromValues(state,action)
        res = p.argMax()
        return res
        # print("PRINTIN UTILCOUNTER")
        # print(p)
        # util.raiseNotDefined()

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
        k = self.iterations
        states = self.mdp.getStates()
        for ki in range(k):
            vector = self.values.copy()
            ind = ki % len(states)
            st= states[ind]
            if self.mdp.isTerminal(st):
                continue
            best = None
            for action in self.mdp.getPossibleActions(st):
                if best is None:
                    best = self.getQValue(st,action)
                else:
                    best = max(best,self.getQValue(st,action))
                vector[st] = best
            self.values=vector 

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
        states = self.mdp.getStates()
        predecessors={}
        pq = util.PriorityQueue()
        self.generatePredecessors(predecessors,states)
        # print(predecessors)
        self.updatePq(pq,states)
        self.iterateOverIterations(pq,predecessors)

    def isTerminal(self,state):
        return self.mdp.isTerminal(state)

    def iterateOverIterations(self,pq,predecessors):
        for ind in range(self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()
            if not self.isTerminal(state):
                self.values[state] = self.computeValue(state)
            for p in predecessors[state]:
                if self.isTerminal(p):
                    continue
                diff = abs(self.values[p]-self.computeValue(p))
                if diff > self.theta:
                    pq.update(p,-diff)

    def computeValue(self,state):
        return max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
    
    def updatePq(self,pq,states):
        for state in states:
            if self.isTerminal(state):
                continue
            maxQVal  = self.computeValue(state)
            diff = abs(self.values[state] -maxQVal)
            pq.update(state,-diff)


    def generatePredecessors(self,predecessors,states):
        for state in states:
            if self.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for pred,_ in self.mdp.getTransitionStatesAndProbs(state,action):
                    if pred not in predecessors:
                        predecessors[pred] = {state}
                    else:
                        predecessors[pred].add(state)


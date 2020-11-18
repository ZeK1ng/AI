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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        # print("Printing food")
        # print(newFood)
        # print("printing pos")
        # print(newPos)
        # print("printing GhostState")
        # print(newGhostStates)
        # print("printing scaredTimes")
        # print(newScaredTimes)
        # print("---------")
        # print(newFood.width)
        dists = list()
        for i in range(newFood.width):
            for j in range(newFood.height):
                if newFood[i][j]:
                    currD = util.manhattanDistance((i,j),newPos)
                    dists.append(currD)
        # print("printing distances")
        # print(dists)
        minDist = 0
        if len(dists) != 0:
            minDist = min(dists)
        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()+1/(minDist+0.00001)

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
        _,res = self.maximi(gameState,0,0)
        return res
            

    def isTerminalState(self,cstate,depth):
        return depth == self.depth or cstate.isWin() or cstate.isLose()

    def minimi(self,state,depth,index):
        if self.isTerminalState(state,depth):
            return self.evaluationFunction(state),None
        can_do_actions = state.getLegalActions(index)
        v=float('inf')
        optimalAction = None
        for action in can_do_actions:
            successor = state.generateSuccessor(index,action)
            if (index +1) % state.getNumAgents() == 0:
                val,_ = self.maximi(successor,depth+1,0)
            else :
                val,_ = self.minimi(successor,depth,index+1)
            if val <v :
                v = val 
                optimalAction = action
        return v,optimalAction

    def maximi(self ,state,depth,index):
        if self.isTerminalState(state,depth):
            return self.evaluationFunction(state),None
        can_do_actions = state.getLegalActions(index)
        v=float('-inf')
        optimalAction = None
        for action in can_do_actions:
            successor = state.generateSuccessor(index,action)
            val,_ = self.minimi(successor,depth,1)
            if val > v:
                v=val
                optimalAction = action
        return v,optimalAction
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        a=float('-inf')
        b=float('inf')
        _,act = self.maximi(gameState,0,0,a,b)
        return act
        
    def isTerminalState(self,cstate,depth):
        return depth == self.depth or cstate.isWin() or cstate.isLose()

    def minimi(self,state,index,depth,a,b):
        if self.isTerminalState(state,depth):
            return self.evaluationFunction(state),None
        v= float('inf')
        can_do_actions = state.getLegalActions(index)
        optimalAction =None
        for action in can_do_actions:
            successor = state.generateSuccessor(index,action)
            if (index+1) % state.getNumAgents() !=0:
                val,_=self.minimi(successor,index+1,depth,a,b)
            else:
                val,_=self.maximi(successor,0,depth+1,a,b)
            if val < v:
                optimalAction = action
            v=min(v,val)
            b=min(v,b)
            if v<a:
                break
        return v , optimalAction

    def maximi(self,state,index,depth, a, b):
        if self.isTerminalState(state,depth):
            return self.evaluationFunction(state),None
        v= float('-inf')
        can_do_actions = state.getLegalActions(index)
        optimalAction = None
        for action in can_do_actions:
            successor = state.generateSuccessor(index,action)
            valu,_=self.minimi(successor,1,depth,a,b)
            if valu > v:
                optimalAction = action
            v=max(v,valu)
            if v>b:
                break
            a = max(a,v)
        return v, optimalAction
        




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"     
        vals = list()
        score_action_map={}
        can_do_actions = gameState.getLegalActions(0)
        for action in can_do_actions:
            successor = gameState.generateSuccessor(0,action)
            val = self.minimi(successor,0,1)
            score_action_map[val]=action
            vals.append(val)
        
        return score_action_map[max(vals)]
            

    def isTerminalState(self,cstate,depth):
        return depth == self.depth or cstate.isWin() or cstate.isLose()

    def minimi(self,state,depth,index):
        if self.isTerminalState(state,depth):
            return self.evaluationFunction(state)
        can_do_actions = state.getLegalActions(index)
        vals = list()
        for action in can_do_actions:
            successor = state.generateSuccessor(index,action)
            if (index +1) % state.getNumAgents() == 0:
                val = self.maximi(successor,depth+1,0)
            else :
                val = self.minimi(successor,depth,index+1)
            vals.append(val)   
        return sum(vals)/len(vals)

    def maximi(self ,state,depth,index):
        if self.isTerminalState(state,depth):
            return self.evaluationFunction(state)
        can_do_actions = state.getLegalActions(index)
        vals= list()
        for action in can_do_actions:
            successor = state.generateSuccessor(index,action)
            val = self.minimi(successor,depth,1)
            vals.append(val)
        return max(vals)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    dists = list()
    for i in range(newFood.width):
        for j in range(newFood.height):
            if newFood[i][j]:
                currD = util.manhattanDistance((i,j),newPos)
                dists.append(currD)
        minDist = 0
        if len(dists) != 0:
            minDist = min(dists)
        "*** YOUR CODE HERE ***"
    ghostList = currentGameState.getGhostPositions()
    ghostDists = list()
    for cord in ghostList:
        ghostDists.append(util.manhattanDistance(list(cord),newPos))
    capsulePos = currentGameState.getCapsules()
    capsList = list()
    for capscord in capsulePos:
        capsList.append(util.manhattanDistance(list(capscord),newPos))
    minDistToCap = 0
    if len(capsList) !=0 :
        minDistToCap = min(capsList)
    return currentGameState.getScore()+1/(minDist+0.00001) - 0.7 *min(ghostDists) + 1.5* 1/(minDistToCap+0.1) #+ 1/(newScaredTimes[0]*min(ghostDists)+0.1)


# Abbreviation
better = betterEvaluationFunction

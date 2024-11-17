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
    visited = set() # Set initialization to keep track of visited states
    stack = util.Stack() # Stack initialization to store the states to be explored
    stack.push((problem.getStartState(), [])) # Push the start state and an empty path onto the stack
    
    # Continue searching while the stack is not empty
    while not stack.isEmpty():
        # Pop the current state and its corresponding path from the stack
        current_state, path = stack.pop()
        
        # If the current state is the goal state, return the path to reach it
        if problem.isGoalState(current_state):
            return path

        # If the current state has not been visited before
        if current_state not in visited:
            # Mark the current state as visited
            visited.add(current_state)
            
            # Explore each unvisited successor of the current state
            for successor in problem.getSuccessors(current_state):
                # Unpack the successor tuple
                new_state, action, cost = successor
                
                # If the successor state has not been visited before
                if new_state not in visited and new_state not in [x[0] for x in stack.list]:
                    # Push the successor state and the updated path onto the stack
                    stack.push((new_state, path + [action]))

    # If no goal state is found and the stack becomes empty, return an empty path
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = set() # Set initialization to keep track of visited states
    queue = util.Queue() # Queue initialization to store the states to be explored
    queue.push((problem.getStartState(), [])) # Push the start state and an empty path onto the queue
    
    # Continue searching while the queue is not empty
    while not queue.isEmpty():
        # Pop the current state and its corresponding path from the queue
        current_state, path = queue.pop()
        
        # If the current state is the goal state, return the path to reach it
        if problem.isGoalState(current_state):
            return path

        # If the current state has not been visited before
        if current_state not in visited:
            # Mark the current state as visited
            visited.add(current_state)
            
            # Explore each unvisited successor of the current state
            for successor in problem.getSuccessors(current_state):
                # Unpack the successor tuple
                new_state, action, cost = successor
                
                # If the successor state has not been visited before
                if new_state not in visited and new_state not in [x[0] for x in queue.list]:
                    # Push the successor state and the updated path onto the queue
                    queue.push((new_state, path + [action]))

    # If no goal state is found and the queue becomes empty, return an empty path
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = set() # Set initialization to keep track of visited states
    priority_queue = util.PriorityQueue() # Priority queue initialization to store the states to be explored
    priority_queue.push((problem.getStartState(), [], 0), 0) # Push the start state, an empty path, and a cost of 0 onto the priority queue
    
    # Continue searching while the priority queue is not empty
    while not priority_queue.isEmpty():
        # Pop the state, path, and cost with the lowest cost from the priority queue
        current_state, path, current_cost = priority_queue.pop()
        
        # If the current state is the goal state, return the path to reach it
        if problem.isGoalState(current_state):
            return path
        
        # If the current state has not been visited before
        if current_state not in visited:
            # Mark the current state as visited
            visited.add(current_state)
            
            # Explore each unvisited successor of the current state
            for successor in problem.getSuccessors(current_state):
                # Unpack the successor tuple
                new_state, action, stepCost = successor
                
                # If the successor state has not been visited before
                if new_state not in visited:
                    # Calculate the new cost to reach the successor
                    newCost = current_cost + stepCost
                    
                    # Push the successor state, updated path, and new cost onto the priority queue
                    priority_queue.push((new_state, path + [action], newCost), newCost)
    
    # If no goal state is found and the priority queue becomes empty, return an empty path
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited = set() # Set initialization to keep track of visited states
    priority_queue = util.PriorityQueue() # Priority queue initialization to store the nodes to be explored
    priority_queue.push((problem.getStartState(), [], 0), 0) # Push the start state, an empty path, and a cost of 0 onto the priority queue
    
    # Continue searching while the priority queue is not empty
    while not priority_queue.isEmpty():
        # Pop the node with the lowest f-score (cost + heuristic) from the priority queue
        current_state, path, current_cost = priority_queue.pop()
        
        # If the current state is the goal state, return the path to reach it
        if problem.isGoalState(current_state):
            return path
        
        # If the current state has not been visited before
        if current_state not in visited:
            # Mark the current state as visited
            visited.add(current_state)
            
            # Explore each unvisited successor of the current state
            for successor in problem.getSuccessors(current_state):
                # Unpack the successor tuple
                new_state, action, stepCost = successor
                
                # If the successor state has not been visited before
                if new_state not in visited:
                    # Calculate the new cost to reach the successor
                    newCost = current_cost + stepCost
                    
                    # Calculate the f-score by adding the cost and heuristic value
                    fScore = newCost + heuristic(new_state, problem)
                    
                    # Push the successor state, updated path, and new cost onto the priority queue
                    priority_queue.push((new_state, path + [action], newCost), fScore)

    # If no goal state is found and the priority queue becomes empty, return an empty path
    return []


#####################################################
# EXTENSIONS TO BASE PROJECT
#####################################################

# Extension Q1e
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE ***"
    
    depth_limit = 0 
    while True:
        stack = util.Stack() # Initialize a stack for storing states
        stack.push((problem.getStartState(), [], 0)) # Push the start state and an empty path and the current depth onto the stack 
        visited = set() # Initialize an empty set to keep track of visited states at each depth
        
        while not stack.isEmpty():
            # Pop the current state, its corresponding path, and depth from the stack
            current_state, path, depth = stack.pop()
            
            # If the current state is the goal state, return the path to reach it
            if problem.isGoalState(current_state):
                return path
            
            # If the current depth exceeds the depth limit, continue to the next iteration
            if depth > depth_limit:
                continue
            
            # If the current state has not been visited at this depth
            if (current_state, depth) not in visited:
                # Mark the current state as visited at this depth
                visited.add((current_state, depth))
                
                # Explore each unvisited successor of the current state
                for successor in problem.getSuccessors(current_state):
                    # Unpack the successor tuple
                    new_state, action, stepCost = successor
                    
                    # Push the successor state, the updated path, and the incremented depth onto the stack
                    stack.push((new_state, path + [action], depth + 1))
        
        # Increment the depth limit for the next iteration
        depth_limit += 1

#####################################################
# Abbreviations
#####################################################
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch

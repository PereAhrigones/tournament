# baseline_team.py
# ---------------
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


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='una polla', second='ReflexCapotraa mastureAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########
class AlphaBetaAgent(CaptureAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function.
        This function initiates the alpha-beta pruning process.
        """
        # Start the alpha-beta pruning process with initial alpha and beta values
        return self.max_value(game_state, 3, -float("inf"), float("inf"))[0]
    
    def max_value(self, game_state, depth, alpha, beta):
        """
        Returns the maximum value for the current game state.
        This function is called for the maximizing player (Pacman).
        """
        # Check if the search has reached the maximum depth or if the game is over
        if depth == self.depth or game_state.is_win() or game_state.is_lose():
            return None, self.evaluation_function(game_state)
        
        max_value = -float("inf")
        max_action = None

        # Iterate over all legal actions for the maximizing player
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            # Call min_value for the minimizing player (ghosts)
            _, value = self.min_value(successor, depth, 1, alpha, beta)
            if value > max_value:
                max_value = value
                max_action = action
            # Alpha-Beta pruning so
            if max_value > beta:
                return max_action, max_value
            alpha = max(alpha, max_value)
        
        return max_action, max_value
    
    def min_value(self, game_state, depth, agent, alpha, beta):
        if depth == self.depth or game_state.is_win() or game_state.is_lose():
            return None, self.evaluation_function(game_state)
        min_value = float("inf")
        min_action = None
        for action in game_state.get_legal_actions(agent):
            successor = game_state.generate_successor(agent, action)
            if agent == game_state.get_num_agents() - 1:
                _, value = self.max_value(successor, depth + 1, alpha, beta)
            else:
                _, value = self.min_value(successor, depth, agent + 1, alpha, beta)
            if value < min_value:
                min_value = value
                min_action = action
            if min_value < alpha:
                return min_action, min_value
            beta = min(beta, min_value)
        return min_action, min_value


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}




"""
Dynamic Programming for FJ-OffDP
--------------------------------

This module implements a classical dynamic programming (policy iteration) algorithm 
for solving the finite Markov Decision Process (MDP) formulation of the 
Offline Depolarization Problem (OffDP) under the Friedkin–Johnsen (FJ) opinion dynamics model.

The environment is provided by `FJOpinionDynamicsFinite`, which enumerates all states 
up to isomorphism. Using policy iteration, the algorithm computes the optimal 
state-value function and policy for reducing polarization by a sequence of 
edge modifications. 

This implementation is only feasible for small graphs, where the full state space 
can be generated and evaluated exhaustively.
"""

from collections import defaultdict
import numpy as np
from env import FJOpinionDynamicsFinite

class DynamicProgramming:
    """
    Policy iteration solver for the finite FJ-OffDP MDP.

    Parameters
    ----------
    env : FJOpinionDynamicsFinite
        Finite environment defining states, actions, and transitions.
    gamma : float, default=1.0
        Discount factor for future rewards.
    theta : float, default=1e-3
        Convergence threshold for policy evaluation.
    V : dict, optional
        Initial value function. Defaults to zero for all states.
    pi : np.ndarray, optional
        Initial policy. Defaults to choosing action 0 for all states.
    """
    def __init__(self, env: FJOpinionDynamicsFinite, gamma: float = 1.0, theta: float = 1e-3, V = None, pi=None):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = V if V is not None else defaultdict(float)
        self.pi = pi if pi is not None else np.zeros(len(env.state_idxes), dtype=int)

    def policy_greedy(self, state):
        """
        Returns the best action for a given state under the current value function.

        Parameters
        ----------
        state : int
            Index of the state in env.states.

        Returns
        -------
        action : int
            Greedy action according to the current policy.
        """
        return self.pi[state]

    def policy_evaluation(self):
        """
        Evaluates the current policy by iteratively applying the Bellman equation.

        Returns
        -------
        V : dict
            State-value function for the current policy.
        """
        delta = float("inf")
        while delta > self.theta:
            delta = 0
            for state in self.env.state_idxes:
                if self.env.is_terminal(state):
                    continue
                a = self.pi[state]
                v_old = self.V[state]
                next_state, reward, _ = self.env.step(a, state)
                self.V[state] = reward + self.gamma * self.V[next_state]
                delta = max(delta, abs(v_old - self.V[state]))

    def policy_improvement(self):
        """
        Improves the current policy by making it greedy with respect to the current value function.

        Returns
        -------
        is_stable : bool
            True if the policy is stable (no changes), False otherwise.
        """
        is_stable = True
        for state in self.env.state_idxes:
            if self.env.is_terminal(state):
                continue
            old_action = self.pi[state]
            best_val = float("-inf")
            best_action = None
            for a in self.env.actions:
                next_state, reward, _ = self.env.step(a, state)
                val = reward + self.gamma * self.V[next_state]
                if val > best_val:
                    best_val = val
                    best_action = a
            self.pi[state] = best_action
            if best_action != old_action:
                is_stable = False
        return is_stable

    def run(self):
        """
        Runs the policy iteration algorithm until convergence.

        Returns
        -------
        V : dict
            Final state-value function.
        pi : np.ndarray
            Final policy.
        """
        stable = False
        while not stable:
            self.policy_evaluation()
            stable = self.policy_improvement()
        return self.V, self.pi

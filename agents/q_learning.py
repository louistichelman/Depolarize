"""
Flat Q-Learning for FJ-OffDP
----------------------------

This module implements a tabular Q-learning algorithm for the finite Markov 
Decision Process (MDP) formulation of the Offline Depolarization Problem (OffDP) 
under the Friedkin–Johnsen (FJ) opinion dynamics model.

Unlike dynamic programming, Q-learning does not require explicit knowledge of 
the transition probabilities and can instead learn directly from simulated 
environment interactions. This makes it a useful stepping stone toward 
scalable reinforcement learning approaches for larger networks.

This implementation is limited to small graph instances, where the state space 
can still be enumerated explicitly.
"""

import numpy as np
from env import FJOpinionDynamicsFinite

class QLearning:
    """
    Tabular Q-learning agent for the finite FJ-OffDP MDP.

    Parameters
    ----------
    env : FJOpinionDynamicsFinite
        Finite environment defining states, actions, and transitions.
    gamma : float, default=1.0
        Discount factor for future rewards.

    """
    def __init__(self, env: FJOpinionDynamicsFinite, gamma: float = 1.0):
        self.env = env
        self.q_table = np.zeros((len(env.states), len(env.actions)))
        self.gamma = gamma

    def policy_greedy(self, state):
        """
        Returns the action with the highest Q-value at the given state.
        """
        return np.argmax(self.q_table[state][:])

    def policy_greedy_epsilon(self, state, epsilon):
        """
        Chooses an action using ε-greedy exploration.
        """
        if np.random.rand() > epsilon:
            return self.policy_greedy(state)
        else:
            return np.random.randint(0, len(self.env.actions))

    def train(
        self,
        n_training_episodes=10000,
        min_epsilon=0.05,
        max_epsilon=1.0,
        learning_rate=0.07,
        take_snapshots_every=None,
    ):
        """
        Train the Q-learning agent using ε-greedy exploration.

        Parameters
        ----------
        n_training_episodes : int, default=10000
            Number of training episodes.
        min_epsilon : float, default=0.05
            Minimum exploration probability.
        max_epsilon : float, default=1.0
            Maximum exploration probability.
        learning_rate : float, default=0.07
            Learning rate for Q-learning updates.
        take_snapshots_every : int, optional
            If specified, take snapshots of the Q-table every N episodes.
        """
        q_table_snapshots = [self.q_table.copy()]
        for episode in range(n_training_episodes):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * episode / n_training_episodes

            state = self.env.reset()

            while True:
                action = self.policy_greedy_epsilon(state, epsilon)
                new_state, reward, terminal = self.env.step(action, state)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.q_table[state][action] = self.q_table[state][
                    action
                ] + learning_rate * (
                    reward
                    + self.gamma * np.max(self.q_table[new_state])
                    - self.q_table[state][action]
                )

                if terminal:
                    break
                state = new_state
            if take_snapshots_every is not None and episode % take_snapshots_every == 0:
                q_table_snapshots.append(self.q_table.copy())
        if take_snapshots_every is not None:
            return self.q_table, q_table_snapshots
        return self.q_table

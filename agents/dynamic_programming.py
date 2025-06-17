from collections import defaultdict
import numpy as np

class DynamicProgramming:
    def __init__(self, env, gamma=1.0, theta=1e-3, V=None, pi=None):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = V if V is not None else defaultdict(float)
        self.pi = pi if pi is not None else np.zeros(len(env.state_idxes), dtype=int)

    def policy_greedy(self, state):
        return self.pi[state]

    def policy_evaluation(self):
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
        stable = False
        while not stable:
            self.policy_evaluation()
            stable = self.policy_improvement()
        return self.V, self.pi

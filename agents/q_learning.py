import numpy as np


class QLearning:
    def __init__(self, env, gamma=1.0):
        self.env = env
        self.q_table = np.zeros((len(env.states), len(env.actions)))
        self.gamma = gamma

    def policy_greedy(self, state):
        return np.argmax(self.q_table[state][:])

    def policy_greedy_epsilon(self, state, epsilon):
        if np.random.rand() > epsilon:
            return self.policy_greedy(state)
        else:
            return np.random.randint(0, len(self.env.actions))

    def train(
        self,
        n_training_episodes=10000,
        min_epsilon=0.05,
        max_epsilon=1.0,
        decay_rate=0.0005,
        learning_rate=0.07,
    ):
        for episode in range(n_training_episodes):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
                -decay_rate * episode
            )

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
        return self.q_table

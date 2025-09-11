import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import wandb
import string

from env import BaseEnv
from .q_network.gnn import GNN_REGISTRY
from .q_network.qnet import QNET_REGISTRY


class DQN:
    """
    Deep Q-Network (DQN) agent for training on graph-based environments.
    This agent uses a GNN architecture for the Q-network and supports various configurations.
    """

    def __init__(
        self,
        env: BaseEnv = None,  # None, if agent only used for evaluation
        gnn: str = "GraphSage",  # has to be in GNN_REGISTRY
        qnet: str = "simple",  # has to be in QNET_REGISTRY
        **kwargs,
    ):

        # Extract from kwargs with fallback defaults
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        learning_rate = kwargs.get("learning_rate", 0.0008)
        self.gamma = kwargs.get("gamma", 1.0)
        self.batch_size = kwargs.get("batch_size", 40)
        self.train_freq = kwargs.get("train_freq", 4)
        self.target_update_freq = kwargs.get("target_update_freq", 1000)
        self.timesteps_train = kwargs.get("timesteps_train", 100000)
        self.wandb_init = kwargs.get("wandb_init", True)
        self.start_e = kwargs.get("start_e", 1.0)
        self.end_e = kwargs.get("end_e", 1.0)
        self.reset_probability = kwargs.get(
            "reset_probability", None
        )  # relevant for non-episodic environments
        self.parallel_envs = kwargs.get("parallel_envs", 1)
        self.td_loss_one_edge = kwargs.get("td_loss_one_edge", False)

        # Create multiple instances of the environment
        if env is not None:
            self.envs = [env.clone() for _ in range(self.parallel_envs)]
            self.n = env.n

        # define architecture
        gnn_architecture = GNN_REGISTRY[gnn]
        qnet_class = QNET_REGISTRY[qnet]

        # Create two identical models for q and target networks
        gnn_q = gnn_architecture(**kwargs)
        gnn_target = gnn_architecture(**kwargs)
        self.q_network = qnet_class(gnn_q)
        self.target_network = qnet_class(gnn_target)

        # copy weights from q_network to target_network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # define optimizer and replay buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = SimpleReplayBuffer(20000)

        # keep track of step even if we train several times
        self.global_step = 0

        if self.wandb_init and env is not None:
            # to ensure a unique run_name we generate a random code
            random_code = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=5)
            )
            num_heads = gnn_q.num_heads if gnn_q.num_heads is not None else 0
            k = env.k if hasattr(env, "k") else 0  # k is not always defined

            self.run_name = kwargs.get(
                "run_name",
                f"{gnn}-{qnet}-n{self.n}-k{k}-hd{gnn_q.embed_dim}-layers{len(gnn_q.layers)}-lr{learning_rate}-heads{num_heads}-bs{self.batch_size}-p{self.parallel_envs}-g{self.gamma}-tuf{self.target_update_freq}-{random_code}",
            )
            wandb.init(
                project="Depolarize",
                name=self.run_name,
                entity="louistichelman",
                config={
                    "architecture": gnn_architecture,
                    "qnet_approach": qnet_class,
                    "embed_dim": gnn_q.embed_dim,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "batch_size": self.batch_size,
                    "gamma": self.gamma,
                    "train_freq": self.train_freq,
                    "target_update_freq": self.target_update_freq,
                    "start_e": self.start_e,
                    "end_e": self.end_e,
                },
            )

    def policy_greedy(self, state, give_q_valaues=False, **kwargs):
        """
        Returns the action with the highest Q-value for the given state.
        """
        with torch.no_grad():
            q_values = self.q_network([state]).squeeze()
        if give_q_valaues:
            return q_values.argmax().item(), q_values
        return q_values.argmax().item()

    def compute_td_loss(self, batch):
        """
        Computes the temporal difference loss for a batch of transitions.
        The batch is a list of tuples (state, action, reward, next_state, done).
        """
        states, actions, rewards, next_states, dones = zip(*batch)

        if self.td_loss_one_edge:
            taus = torch.tensor(
                [state["tau"] is None for state in states],
                dtype=torch.float32,
                device=self.device,
            )
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q-values per node in batch
        q_values_all = self.q_network(states).squeeze()  # shape: [B * n]
        next_q_values_all = self.target_network(next_states).squeeze()  # shape: [B * n]

        q_values = q_values_all.view(len(batch), self.n)  # Reshape to [B, n]
        next_q_values = next_q_values_all.view(len(batch), self.n)  # Reshape to [B, n]

        # Select Q-values for the actions taken
        q_selected = q_values.gather(1, actions.view(-1, 1)).squeeze()  # shape: [B]

        # Get the maximum Q-value for the next states
        max_next_q = next_q_values.max(dim=1).values

        if self.td_loss_one_edge:
            # Compute target: reward for adding one edge
            td_target = rewards + taus * max_next_q
        else:
            # Compute target: reward + gamma * max_a' Q(s', a') * (1 - done)
            td_target = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.mse_loss(q_selected, td_target)
        return loss

    @staticmethod
    def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        """
        Linear schedule for epsilon-greedy exploration.
        """
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def train(self, profiler=None):
        """
        Trains the DQN agent.
        """
        # if we have multiple parallel environments, we keep track of a list of current states, instead of a single state
        states = [self.envs[i].reset() for i in range(self.parallel_envs)]

        for step in range(1, self.timesteps_train + 1):
            self.global_step += 1
            epsilon = self.linear_schedule(
                self.start_e, self.end_e, self.timesteps_train, step
            )
            if random.random() < epsilon:  # epsilon-greedy exploration
                actions = [
                    random.randint(0, self.n - 1) for _ in range(self.parallel_envs)
                ]
            else:
                q_values = self.q_network(states).squeeze()  # Shape: [B*n]
                q_values = q_values.view(len(states), self.n)  # Reshape to [B, n]
                actions = q_values.argmax(
                    dim=1
                ).tolist()  # decide on action with highest Q-value

            next_states = []
            for i in range(self.parallel_envs):
                next_state, reward, done = self.envs[i].step(actions[i])
                self.replay_buffer.add(states[i], actions[i], reward, next_state, done)
                next_state = next_state if not done else self.envs[i].reset()
                if (  # for non-episodic environments, we reset the environment with a certain probability
                    self.reset_probability is not None
                    and random.random() < self.reset_probability
                ):
                    next_state = self.envs[i].reset()
                next_states.append(next_state)

            states = next_states

            if (
                len(self.replay_buffer) >= self.batch_size
                and step % self.train_freq == 0
            ):
                batch = self.replay_buffer.sample(self.batch_size)
                loss = self.compute_td_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.wandb_init:
                    wandb.log(
                        {
                            "step": self.global_step,
                            "loss": loss.item(),
                            "epsilon": epsilon,
                        }
                    )
                if profiler is not None:
                    profiler.step()

            if step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        if self.wandb_init:
            wandb.finish()


class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

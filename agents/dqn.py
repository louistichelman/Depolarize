import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import wandb
import string
from visualization import visualize_graph, visualize_opinions
from .q_network.gnn_architectures import ARCHITECTURE_REGISTRY
from .q_network.qnet_approaches import QNET_REGISTRY
import os

        
class DQN:
    def __init__(self, 
                 env, 
                 model_architecture: str,      #has to be in ARCHITECTURE_REGISTRY
                 qnet_approach: str,           #has to be in QNET_REGISTRY
                 **kwargs):

        # Extract from kwargs with fallback defaults
        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        learning_rate = kwargs.get("learning_rate", 0.0008)
        self.gamma = kwargs.get("gamma", 1.0)
        self.batch_size = kwargs.get("batch_size", 40)
        self.train_freq = kwargs.get("train_freq", 4)
        self.target_update_freq = kwargs.get("target_update_freq", 1000)
        self.timesteps_train = kwargs.get("timesteps_train", 100000)
        self.wandb_init = kwargs.get("wandb_init", True)
        self.start_e = kwargs.get("start_e", 1.0)
        self.end_e = kwargs.get("end_e", 1.0)
        self.reset_probability = kwargs.get("reset_probability", None)
        self.parallel_envs = kwargs.get("parallel_envs", 1)  # Number of parallel environments, default is 1
        self.log_step = kwargs.get("log_step", 10000)

        self.envs = [env.clone() for _ in range(self.parallel_envs)]  # Create multiple instances of the environment
        self.n = env.n

        model_class = ARCHITECTURE_REGISTRY[model_architecture]
        qnet_class = QNET_REGISTRY[qnet_approach]

        model_q = model_class(graph_size_training = self.n, **kwargs) # Create two identical models for q and target networks
        model_target = model_class(graph_size_training = self.n, **kwargs)

        self.q_network = qnet_class(model_q)
        self.target_network = qnet_class(model_target)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = SimpleReplayBuffer(80000)

        self.global_step = 0

        if self.wandb_init:
            random_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) # to ensure a unique run_name
            embed_dim = model_q.embed_dim
            num_layers = len(model_q.layers)
            num_heads = model_q.num_heads if model_q.num_heads is not None else 0
            if kwargs.get("environment") == "friedkin_johnson":
                self.run_name = kwargs.get("run_name", f"{model_architecture}-{qnet_approach}-n{self.n}-k{env.k}-hd{embed_dim}-layers{num_layers}-lr{learning_rate}-heads{num_heads}-bs{self.batch_size}-{random_code}")
            else:
                self.run_name = kwargs.get("run_name", f"{model_architecture}-{qnet_approach}-n{self.n}-hd{embed_dim}-layers{num_layers}-lr{learning_rate}-heads{num_heads}-bs{self.batch_size}-g{self.gamma}-par{self.parallel_envs}-e{self.end_e}-tuf{self.target_update_freq}-{random_code}")
            wandb.init(
            project="Depolarize",
            name=self.run_name,
            entity="louistichelman",
            config={
                "architecture": model_architecture,
                "qnet_approach": qnet_approach,
                "embed_dim": embed_dim,
                "lr": self.optimizer.param_groups[0]['lr'],
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "train_freq": self.train_freq,
                "target_update_freq": self.target_update_freq,
                "start_e": self.start_e,
                "end_e": self.end_e,
            })
        
    def policy_greedy(self, state):
        """
        Returns the action with the highest Q-value for the given state.
        """
        with torch.no_grad():
            q_values = self.q_network([state]).squeeze()
        return q_values.argmax().item()
    
    
    def compute_td_loss(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q-values per node (shape: [total_nodes, 1])
        q_values_all = self.q_network(states).squeeze()            # shape: [B * n]
        next_q_values_all = self.target_network(next_states).squeeze()  # shape: [B * n]

        q_values = q_values_all.view(len(batch), self.n)
        next_q_values = next_q_values_all.view(len(batch), self.n)

        # Select Q-values for the actions taken (shape: [B])
        q_selected = q_values.gather(1, actions.view(-1, 1)).squeeze()

        # Compute target: reward + gamma * max_a' Q(s', a') * (1 - done)
        max_next_q = next_q_values.max(dim=1).values
        td_target = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.mse_loss(q_selected, td_target)
        return loss

    @staticmethod
    def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)
    
    def train(self, profiler = None):
        states = [self.envs[i].reset() for i in range(self.parallel_envs)]
        loss_window = deque(maxlen=100)
        opinions = [states[0]["sigma"].copy()]
        for step in range(1, self.timesteps_train + 1):
            self.global_step += 1
            epsilon = self.linear_schedule(self.start_e, self.end_e, self.timesteps_train, step)
            if random.random() < epsilon:
                actions = [random.randint(0, self.n - 1) for _ in range(self.parallel_envs)]
            else:
                q_values = self.q_network(states).squeeze()  # Shape: [B*n]
                q_values = q_values.view(len(states), self.n)
                actions = q_values.argmax(dim=1).tolist()

            next_states = []
            for i in range(self.parallel_envs):
                next_state, reward, done = self.envs[i].step(actions[i])
                # tau = states[i]["tau"]
                # if tau is not None:
                    # print(f"tau: {tau}, opinion: {states[i]['sigma'][tau]}")
                    # print(f"action: {actions[i]}, opinion: {states[i]['sigma'][actions[i]]}")
                    # print("connected") if states[i]["graph"].has_edge(tau, actions[i]) else print("not connected")
                    # print(reward)
                self.replay_buffer.add(states[i], actions[i], reward, next_state, done)
                next_state = next_state if not done else self.envs[i].reset()
                if self.reset_probability is not None and random.random() < self.reset_probability:
                    next_state = self.envs[i].reset()
                next_states.append(next_state)

            states = next_states
            opinions.append(states[0]["sigma"].copy())

            if step % self.log_step == 0:
                run_dir = os.path.join("saved files", "dqn", "nonlinear", "saved_runs", self.run_name, "logging")
                os.makedirs(run_dir, exist_ok=True)
                visualize_graph(states[0]["graph"], states[0]["sigma"], title=f"Step {step}", file_path=os.path.join(run_dir, f"step_{step}_graph.png"))
                visualize_opinions(self.envs[0], opinions, title=f"Step {step}", file_path=os.path.join(run_dir, f"step_{step}_opinions.png"))
            
            # if self.reset_env_after_n_steps is not None and step % self.reset_env_after_n_steps == 0:
            #     states = [self.envs[i].reset() for i in range(self.parallel_envs)]
            #     opinions = [states[0]["sigma"].copy()]

            if len(self.replay_buffer) >= self.batch_size and step % self.train_freq == 0:
                batch = self.replay_buffer.sample(self.batch_size)
                loss = self.compute_td_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_window.append(loss.item())
                if self.wandb_init:
                    wandb.log({"step": self.global_step, "loss": loss.item(), "epsilon": epsilon})
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
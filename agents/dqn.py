import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import wandb
import string
from .q_network.gnn_architectures import ARCHITECTURE_REGISTRY
from .q_network.qnet_approaches import QNET_REGISTRY

        
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
        self.exploration_fraction = kwargs.get("exploration_fraction", 0.1)

        self.env = env

        model_class = ARCHITECTURE_REGISTRY[model_architecture]
        qnet_class = QNET_REGISTRY[qnet_approach]

        model_q = model_class(graph_size_training = env.n, **kwargs) # Create two identical models for q and target networks
        model_target = model_class(graph_size_training = env.n, **kwargs)

        self.q_network = qnet_class(model_q)
        self.target_network = qnet_class(model_target)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = SimpleReplayBuffer(10000)

        self.global_step = 0

        if self.wandb_init:
            random_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) # to ensure a unique run_name
            embed_dim = kwargs.get("num_heads", "_default")
            num_layers = len(model_q.layers)
            num_heads = kwargs.get("num_heads", "_default")
            self.run_name = kwargs.get("run_name", f"{model_architecture}-{qnet_approach}-n{env.n}-k{env.k}-hd{embed_dim}-layers{num_layers}-lr{learning_rate}-heads{num_heads}-bs{self.batch_size}-{random_code}")
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
                "exploration_fraction": self.exploration_fraction,
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

        n = self.env.n 
        q_values = q_values_all.view(len(batch), n)
        next_q_values = next_q_values_all.view(len(batch), n)

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
        state = self.env.reset()
        loss_window = deque(maxlen=100)
        for step in range(1, self.timesteps_train + 1):
            self.global_step += 1
            epsilon = self.linear_schedule(self.start_e, self.end_e, self.exploration_fraction * self.timesteps_train, step)
            if random.random() < epsilon:
                action = random.randint(0, self.env.n - 1)
            else:
                q_values = self.q_network([state]).squeeze()
                action = q_values.argmax().item()

            next_state, reward, done = self.env.step(action)
            self.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state if not done else self.env.reset()

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
            
class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
# Reducing Opinion Polarization in Social Networks via Edge Modifications and Graph Neural Networks

This repository contains the code accompanying my thesis:  
**"Reducing Opinion Polarization in Social Networks via Edge Modifications and Graph Neural Networks"**

The project implements and evaluates **heuristic** and **reinforcement learning approaches** for the problem of reducing opinion polarization in social networks through edge modifications.

---

## Overview of Experiments

The code supports the following experiments:

- **Greedy heuristics (Chapter 3.3):**  
  Implementation of greedy edge modification strategies for the **Offline Depolarization Problem** under Friedkin–Johnson (FJ-OffDP). Includes comparison to optimal solutions on small graphs (see Table 3.1).

- **Nonlinear opinion dynamics (Chapter 4.1):**  
  Implementation of a nonlinear opinion dynamics model and intuitive baseline strategies for the **Online Depolarization Problem** (NL-OnDP).

- **Classical RL approaches (Chapter 5.2):**  
  Non-deep RL methods (dynamic programming and tabular Q-learning) for solving FJ-OffDP.

- **Deep RL approaches (Chapters 5.3 & 5.4):**  
  DQN agents with graph neural networks (GraphSAGE, Graphormer, etc.) for both FJ-OffDP and NL-OnDP.

---

## Usage

All scripts are under `scripts/` and numbered according to their place in the experimental workflow.

### 1. Compare heuristics and optimal solutions (FJ-OffDP)

Run greedy heuristics and compare them to the optimal solutions from Rász et al. on small graphs:

```bash
python scripts/1_compare_heuristics.py \
     --num_graphs 20 \         # number of random graphs to generate
     --num_nodes 10 \          # number of nodes in each graph
     --k 3 \                   # number of edges to modify
     --edge_prob 0.4           # edge probability for random graph generation
```

### 2. Run Dynamic Programming and Q-learning (FJ-OffDP)

Run non-deep RL methods for small instances:

```bash
python scripts/2_run_dp_and_q_learning.py \
     --n_nodes 5 \             # number of nodes in the graph
     --k_steps 2 \             # number of steps (edge modifications)
     --max_edges 5 \           # maximum number of edges in the graph
     --training_episodes 40000 \ # number of Q-learning training episodes
     --learning_rate 1.0 \     # learning rate for Q-learning
     --snapshots_qlearning 5000 # interval for saving Q-learning snapshots
```

### 3. DQN experiments (FJ-OffDP)

#### (i) Create datasets

```bash
python scripts/3_make_data_set.py \
     --env friedkin-johnson \  # environment type (friedkin-johnson for FJ-OffDP)
     --n 150 \                 # graph size for training
     --n_train 1000 \          # number of training graphs
     --n_val 100 \             # number of validation graphs
     --n_test 100 \            # number of test graphs
     --out_of_distribution_n 100 200 300 400 \ # graph sizes for OOD validation/test sets
```

Graphs are saved under `data/friedkin-johnson/`.

#### (ii) Compute greedy solutions

```bash
python scripts/4_compute_greedy_solutions.py \
     --n_values 100 150 200 300 400 \ # graph sizes to evaluate greedy solutions on
     --k_values 5 10 15 20 \          # values of k (edge modifications) to test
     --folder val                     # dataset to use (val or test)
```

Greedy solutions are saved under `data/friedkin-johnson/greedy_solutions`.

#### (iii) Train a DQN agent

```bash
python scripts/7_train.py \
     --environment friedkin-johnson \ # environment type (FJ-OffDP)
     --n 150 \                        # graph size for training
     --k 15 \                         # number of edges to modify during training
     --wandb_init \                   # enable logging to Weights & Biases
     --gnn GraphSage \                # GNN architecture (GraphSage, GraphormerGD, GlobalMP, GCN)
     --qnet CE \                      # Q-network type (CE or DP)
     --learning_rate 0.0004 \         # learning rate for optimizer
     --embed_dim 128 \                # embedding dimension of GNN layers
     --num_layers 4 \                 # number of GNN layers
     --batch_size 64 \                # batch size for training
     --gamma 1.0 \                    # discount factor
     --num_heads 4 \                  # number of attention heads (for GraphormerGD)
     --end_e 1.0 \                    # final epsilon for epsilon-greedy exploration
     --td_loss_one_edge \             # maximize reward per edge instead of per episode
     --train_freq 4 \                 # frequency of training updates
     --target_update_freq 100000 \    # frequency of target network updates
     --timesteps_train 300000 \       # total number of training timesteps
     --n_values 100 150 200 300 400 \ # graph sizes for evaluation (OOD)
     --k_values 10 15 20 25 \         # k values for evaluation
     --folder val \                   # dataset split for evaluation
     --number_of_reruns 3             # (optional) number of independent reruns
```

Trained models and evaluations are saved under `results/dqn/friedkin-johnson/runs/<run_name>`.

#### (iv) Optional: Continue or rerun training

Continue training from a saved run:
```bash
python scripts/7_train.py \
     --run_name <run_name> \          # name of the run folder (from results/friedkin-johnson/runs)
     --timesteps_train 300000 \       # additional timesteps to train
     --n_values 100 150 200 300 400 \ # evaluation graph sizes
     --k_values 10 15 20 25 \         # evaluation k values
     --folder val                     # dataset split for evaluation
```

Rerun a training multiple times:
```bash
python scripts/7_train.py \
     --run_name <run_name> \          # name of the run folder
     --number_of_reruns 3 \           # number of reruns
     --n_values 100 150 200 300 400 \ # evaluation graph sizes
     --k_values 10 15 20 25 \         # evaluation k values
     --folder val                     # dataset split for evaluation
```

### 4. DQN experiments (NL-OnDP)

#### (i) Create datasets

```bash
python scripts/3_make_data_set.py \
     --env nonlinear \                # environment type (nonlinear model for NL-OnDP)
     --n 150 \                        # graph size for training
     --n_train 100 \                  # number of training graphs
     --n_val 10 \                     # number of validation graphs
     --n_test 10 \                    # number of test graphs
     --out_of_distribution_n 100 200 300 \ # graph sizes for OOD validation/test sets
```

Graphs are saved under `data/nonlinear/`.

#### (ii) Compute baselines

```bash
python scripts/5_evaluate_baseline_strategies.py \
     --n_values 100 150 200 300 \     # graph sizes for evaluation
     --n_steps 20000 \                # number of simulation steps
     --n_edge_updates_per_step 4 \    # number of edge updates per step in nonlinear dynamics
     --folder val                     # dataset split (val or test)
```

Baseline results and visualizations are saved under `data/nonlinear/baselines/`.

#### (iii) Optional: Create enhanced training set (Training Method 1 in Chapter 5.4)

```bash
python scripts/6_make_diverse_training_set.py \
     --env nonlinear \                # environment type (NL-OnDP)
     --n 150 \                        # graph size for training
     --average_degree 6 \             # average node degree in graphs
     --n_edge_updates_per_step 4 \    # number of edge updates per step in nonlinear dynamics
     --start_time_values 0 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 \
                                      # time offsets for setting off dynamics (generates diverse states)
```

#### (iv) Train a DQN agent

```bash
python scripts/7_train.py \
    --environment nonlinear \         # environment type (nonlinear for NL-OnDP)
    --n 150 \                         # graph size for training
    --n_edge_updates_per_step 4 \     # number of edge updates per step in nonlinear dynamics
    --wandb_init \                    # enable logging to Weights & Biases
    --gnn GraphSage \                 # GNN architecture
    --qnet CE \                       # Q-network type
    --learning_rate 0.0001 \          # learning rate
    --embed_dim 128 \                 # embedding dimension
    --num_layers 4 \                  # number of GNN layers
    --batch_size 64 \                 # batch size
    --gamma 0.5 \                     # discount factor
    --num_heads 4 \                   # number of attention heads
    --reset_probability 0.02 \        # probability of resetting env after each interaction
    --parallel_envs 1 \               # number of parallel environments (used for Training Method 2)
    --use_diverse_start_states \      # use diverse starting states (from script 6)
    --end_e 0.4 \                     # final epsilon for epsilon-greedy exploration
    --train_freq 8 \                  # frequency of training updates
    --target_update_freq 40000 \      # frequency of target network updates
    --timesteps_train 400000 \        # number of training timesteps
    --n_values 100 150 200 300 \      # evaluation graph sizes
    --folder val \                    # dataset split for evaluation
    --keep_influence_matrix \         # keep influence matrix in states, MUST be set to true for GraphormerGD training (per default fundamental matrix is used, if not --keep_resistance_matrix)
    --keep_resistance_matrix \        # keep resistance matrix in states (--keep_influence_matrix must also set to true!)
    --record_opinions_while_training  # record opinion trajectories during training
```

Trained models and evaluations are saved under `results/dqn/nonlinear/runs/<run_name>`. Again, training can be continued or rerun as described in 4(iv).


## Repository Structure

- `scripts/`: runnable experiment scripts (1–7)
- `env/`: environments for FJ and NL opinion dynamics
- `agents/`: reinforcement learning agents (DP, Q-learning, DQN with GNNs)
- `evaluation/`: evaluation utilities (baselines, DQN evaluation)
- `visualization/`: plotting and analysis utilities
- `data/`: generated datasets and baseline solutions
- `results/`: trained models and evaluation outputs

---

## Notes

- Experiments are logged with [Weights & Biases](https://wandb.ai) if `--wandb_init` is enabled.  
- Run names are auto-generated based on hyperparameters.  
- All models and evaluation outputs are reproducible via the provided scripts.  

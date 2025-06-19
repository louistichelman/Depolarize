import os
import pickle
from env.fj_depolarize import FJDepolarize
from env.fj_depolarize_simple import DepolarizeSimple
from depolarizing_functions import depolarize_greedy
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import os

def generate_random_states(ns, ks, num_states=500, save_path="saved files/dqn/greedy_comparison"):
    os.makedirs(save_path, exist_ok=True)
    all_states = {}
    print("halleo")

    for n in ns:
        for k in ks:
            env = FJDepolarize(n=n, k=k)
            env_simple = DepolarizeSimple(n=n, k=k)
            states = []
            seen = set()
            while len(states)<500:
                state = env.reset()
                state_hash = env_simple.state_hash(state["graph"], state["sigma"], state["tau"])
                if state_hash in seen:
                    if three_times:
                        break
                    if two_times:
                        three_times = True
                    if one_times:
                        two_times = True
                    one_times = True
                    continue
                one_times = two_times = three_times = False
                seen.add(state_hash)
                states.append(state)
            all_states[(n, k)] = states

    with open(os.path.join(save_path, "test_states.pkl"), "wb") as f:
        pickle.dump(all_states, f)

    return all_states

def compute_greedy_polarizations(state_dict, save_path="saved files/dqn/greedy_comparison"):
    os.makedirs(save_path, exist_ok=True)
    results = {}

    for (n, k), states in state_dict.items():
        env = FJDepolarize(n=n, k=k)
        polarizations = []
        for state in states:
            _, pol = depolarize_greedy(state, env)
            polarizations.append(pol)
        results[(n, k)] = polarizations

    with open(os.path.join(save_path, "greedy_polarizations.pkl"), "wb") as f:
        pickle.dump(results, f)

    return results

def compute_greedy_for_state(state_data):
    n, k, i, state = state_data
    env = FJDepolarize(n=n, k=k)
    _, pol = depolarize_greedy(state, env)
    return (n, k, i, pol)

def compute_greedy_polarizations_parallel(state_dict, max_workers=8, save_path="saved files/dqn/greedy_comparison"):
    tasks = []

    # Prepare input tuples (n, k, index, state) for parallel processing
    for (n, k), states in state_dict.items():
        for i, state in enumerate(states):
            tasks.append((n, k, i, state))

    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_state = {executor.submit(compute_greedy_for_state, t): t for t in tasks}
        for future in as_completed(future_to_state):
            n, k, i, pol = future.result()
            if (n, k) not in results:
                results[(n, k)] = {}
            results[(n, k)][i] = pol

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "greedy_polarizations_parallel.pkl"), "wb") as f:
        pickle.dump(results, f)

    return results


import networkx as nx
import matplotlib.pyplot as plt

def visualize_tracetory(start_state, env, policy):
    current_state = start_state
    terminal = env.is_terminal(current_state)
    while True:
        if isinstance(current_state, tuple):
            G, sigma, tau, l = current_state
        else:
            G, sigma, tau, l = env.states[current_state]
        if tau is None:
            polarization = env.polarization(G, sigma)
            pos = nx.spring_layout(G)  # Generate positions for visualization
            color_map = ['red' if sigma[node]==1 else 'blue' for node in G.nodes()]
            plt.figure(figsize=(3, 3))
            nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=500, font_color='white')
            plt.title(f"time_step = {l}, polarization = {polarization}") 
        if terminal:
            break
        action = policy(current_state)
        next_state, _, terminal = env.step(action, current_state)
        current_state = next_state
from collections import deque
import os
import random

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn

from DQN import DQN, Dueling_DQN
from problem_setup import Maze

walls = [
    (1, 3),
    (2, 2), (2, 3), (2, 4), (2, 5), (2, 7),
    (3, 3),
    (4, 5),
    (5, 0), (5, 1), (5, 2), (5, 5),
    (6, 5),
    (7, 5)
]

goal = (0, 5)

start = (6, 1)

yellows = [
    (0, 1),
    (2, 6),
    (4, 1), (4, 6)
]
reds = [
    (2, 1),
    (5, 3),
    (6, 2), (6, 6)
]

def select_action(state, q_network, epsilon):
    """
    state: (row, col) tuple
    q_network: the Q-network
    epsilon: current exploration rate
    returns: action index (0=←, 1=→, 2=↑, 3=↓)
    """
    if np.random.random() < epsilon:
        # Random action
        return np.random.randint(4)
    else:
        # Greedy action
        state_tensor = torch.FloatTensor([state[0], state[1]])
        with torch.no_grad():
            q_values = q_network(state_tensor)
        return q_values.argmax().item()

def train_DQN(maze, q_network, target_network, D_size=10000, num_episodes=1000, α=0.01, γ=0.99, ε=0.1, p=0.025, N_batch=64, N_QU=10, T_epi=100, η=0.01):
    D = deque(maxlen=D_size)  # replay memory
    optimizer = torch.optim.Adam(q_network.parameters(), lr=α) # Using Adam optimzer for computing gradient as recommended

    Epi_Rewards = []
    Epi_Losses = []
    Epi_Lengths = []

    for episode in range(num_episodes):
        episode_reward = 0
        episode_loss = 0
        episode_length = 0
        # Start with a random state
        current_state = maze.random_initial_state() # start with a random state (not necessarily the start state)

        for t in range(T_epi):  # T_epi = 50
            action = select_action(current_state, q_network, ε)  # Implement epsilon-greedy action selection

            next_state, reward, done = maze.step(current_state, action, p) #Take the next step according to the greedy choice
            
            D.append((current_state, action, next_state, reward, done)) # Implement DQN training loop here

            episode_reward += reward
            episode_length += 1
            # select, step, store, update
            if done:
                print(f"Episode finished after {episode_length} steps with reward {episode_reward:.2f}")
                break

            current_state = next_state

            # Every N_QU steps, update Q_w
            if t % N_QU == 0 and len(D) >= N_batch:
                # Taking a random batch from D
                batch = random.sample(D, N_batch)
                states, actions, next_states, rewards, dones = zip(*batch)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    max_next_q = target_network(next_states).max(1)[0]
                    targets = rewards + γ * max_next_q * (1 - dones)

                # compute loss and update q_network parameters
                loss = nn.MSELoss()(current_q, targets)
                episode_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for param, target_param in zip(q_network.parameters(), target_network.parameters()):
                    target_param.data.copy_(η * param.data + (1 - η) * target_param.data)

        ε = max(0.1, ε * 0.99)  # Decay epsilon
        Epi_Rewards.append(episode_reward)
        Epi_Losses.append(episode_loss)
        Epi_Lengths.append(episode_length)

    return Epi_Rewards, Epi_Losses, Epi_Lengths

def moving_average(data, m=25):
    avg = []
    for k in range(len(data)):
        m_k = min(m, k+1)
        avg.append(np.mean(data[max(0, k-m_k+1):k+1]))
    return avg

def plot_training_rewards_and_losses(Avg_Rewards, Avg_Losses, question='q1', scenario='standard_DQN'):
    # Plot Average Reward
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(Avg_Rewards)), Avg_Rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.set_title('Average Reward vs Episode')
    plt.tight_layout()
    os.makedirs(f'{question}/{scenario}', exist_ok=True)
    plt.savefig(f'{question}/{scenario}/avg_reward.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot Average Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(Avg_Losses)), Avg_Losses)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Loss')
    ax.set_title('Average Loss vs Episode')
    plt.tight_layout()
    plt.savefig(f'{question}/{scenario}/avg_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_avg_length(Avg_Lengths, question='q6', scenario='standard_DQN'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(Avg_Lengths)), Avg_Lengths)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Length')
    ax.set_title('Average Episode Length vs Episode')
    plt.tight_layout()
    os.makedirs(f'{question}/{scenario}', exist_ok=True)
    plt.savefig(f'{question}/{scenario}/avg_length.png', dpi=150, bbox_inches='tight')
    plt.close()

def run_q7(maze):
    results = {}
    
    for alpha in [0.0001, 0.001, 0.1]:
        q_network = DQN()
        target_network = DQN()
        target_network.load_state_dict(q_network.state_dict())
        
        Epi_Rewards, _, _ = train_DQN(
            maze, q_network, target_network,
            D_size=10000, 
            num_episodes=750, α=alpha, γ =0.99, 
            ε=1.0, p=0.025, N_batch=64, 
            N_QU=4, T_epi=50, η=0.01
        )
        results[alpha] = moving_average(Epi_Rewards)
    
    # Plot all three on one figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results[0.0001], label='α=0.0001')
    ax.plot(results[0.001], label='α=0.001')
    ax.plot(results[0.1], label='α=0.1')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.set_title('Avg Reward vs Episode for Different Learning Rates')
    ax.legend()
    plt.tight_layout()
    os.makedirs('q7/standard_DQN', exist_ok=True)
    plt.savefig('q7/standard_DQN/avg_reward_lr_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def train_double_DQN(maze, q_network, target_network, D_size=10000, num_episodes=750, alpha=0.001, gamma=0.99, ε=1.0, p=0.025, N_batch=64, N_QU=4, T_epi=50, η=0.01):
    D = deque(maxlen=D_size)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=alpha)

    Epi_Rewards = []
    Epi_Losses = []
    Epi_Lengths = []

    for episode in range(num_episodes):
        episode_reward = 0
        episode_loss = 0
        episode_length = 0
        current_state = maze.random_initial_state()

        for t in range(T_epi):
            action = select_action(current_state, q_network, ε)
            next_state, reward, done = maze.step(current_state, action, p)
            
            D.append((current_state, action, next_state, reward, done))
            episode_reward += reward
            episode_length += 1

            if done:
                break

            if t % N_QU == 0 and len(D) >= N_batch:
                batch = random.sample(D, N_batch)
                states, actions, next_states, rewards, dones = zip(*batch)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_actions = q_network(next_states).argmax(1, keepdim=True)
                    next_q = target_network(next_states).gather(1, next_actions).squeeze(1)
                    targets = rewards + gamma * next_q * (1 - dones)

                loss = nn.MSELoss()(current_q, targets)
                episode_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for param, target_param in zip(q_network.parameters(), target_network.parameters()):
                    target_param.data.copy_(η * param.data + (1 - η) * target_param.data)

            current_state = next_state

        ε = max(0.1, ε * 0.99)
        Epi_Rewards.append(episode_reward)
        Epi_Losses.append(episode_loss)
        Epi_Lengths.append(episode_length)

    return Epi_Rewards, Epi_Losses, Epi_Lengths

def run_problem():
    print("Running DQN problem...")

    maze = Maze(length=8, width=8, walls=walls, yellows=yellows, reds=reds, goal=goal, start=start, penalties={'wall': -0.8, 'red': -10, 'yellow': -5, 'goal': 100})
    q_network = DQN()
    target_network = DQN()
    target_network.load_state_dict(q_network.state_dict())

    Epi_Rewards, Epi_Losses, Epi_Lengths = train_DQN(maze, q_network, target_network, 
                D_size=10000, 
                num_episodes=750, α=0.001, γ =0.99, 
                ε=1.0, p=0.025, N_batch=64, 
                N_QU=4, T_epi=50, η=0.01)    

    Avg_Rewards = moving_average(Epi_Rewards)
    Avg_Losses = moving_average(Epi_Losses)
    Avg_Lengths = moving_average(Epi_Lengths)

    print("Answering q.2...")
    plot_training_rewards_and_losses(Avg_Rewards, Avg_Losses, question='q1', scenario='standard_DQN')

    print("Answering q.3...")
    optimal_policy = maze.get_policy(q_network)
    maze.print_board([optimal_policy], scenario='standard_DQN', type='optimal_policy', question='q3')

    print("Answering q.4...")
    values = maze.get_state_values(q_network)
    maze.print_board([values], scenario='standard_DQN', type='values', question='q4')

    print("Answering q.5...")
    optimal_path = maze.get_path(q_network)
    maze.print_board(trajectories=[optimal_path], scenario='standard_DQN', type='optimal_path', question='q5')

    print("Answering q.6...")
    plot_avg_length(Avg_Lengths)

    print("Answering q.7...")
    run_q7(maze)

    print("Implementing Double DQN...")
    # Double DQN
    q_network_double = DQN()
    target_network_double = DQN()
    target_network_double.load_state_dict(q_network_double.state_dict())

    Epi_Rewards_double, Epi_Losses_double, Epi_Lengths_double = train_double_DQN(
        maze, q_network_double, target_network_double
    )

    Avg_Rewards_double = moving_average(Epi_Rewards_double)
    Avg_Losses_double = moving_average(Epi_Losses_double)
    Avg_Lengths_double = moving_average(Epi_Lengths_double)

    plot_training_rewards_and_losses(Avg_Rewards_double, Avg_Losses_double, question='q8', scenario='double_DQN')
    plot_avg_length(Avg_Lengths_double, question='q8', scenario='double_DQN')

    policy_double = maze.get_policy(q_network_double)
    maze.print_board(boards=[policy_double], scenario='double_DQN', type='policy', question='q8')

    values_double = maze.get_state_values(q_network_double)
    maze.print_board(boards=[values_double], scenario='double_DQN', type='values', question='q8')

    path_double = maze.get_path(q_network_double)
    maze.print_board(trajectories=[path_double], scenario='double_DQN', type='path', question='q8')

    print("Implementing Dueling DQN...")
    # Dueling DQN
    q_network_dueling = Dueling_DQN()
    target_network_dueling = Dueling_DQN()
    target_network_dueling.load_state_dict(q_network_dueling.state_dict())

    Epi_Rewards_dueling, Epi_Losses_dueling, Epi_Lengths_dueling = train_DQN(
                        maze, q_network_dueling, target_network_dueling,
                        D_size=10000,
                        num_episodes=750, α=0.001, γ=0.99,
                        ε=1.0, p=0.025, N_batch=64,
                        N_QU=4, T_epi=50, η=0.01
                    )

    Avg_Rewards_dueling = moving_average(Epi_Rewards_dueling)
    Avg_Losses_dueling = moving_average(Epi_Losses_dueling)
    Avg_Lengths_dueling = moving_average(Epi_Lengths_dueling)

    plot_training_rewards_and_losses(Avg_Rewards_dueling, Avg_Losses_dueling, question='q9', scenario='dueling_DQN')
    plot_avg_length(Avg_Lengths_dueling, question='q9', scenario='dueling_DQN')

    policy_dueling = maze.get_policy(q_network_dueling)
    maze.print_board(boards=[policy_dueling], scenario='dueling_DQN', type='policy', question='q9')

    values_dueling = maze.get_state_values(q_network_dueling)
    maze.print_board(boards=[values_dueling], scenario='dueling_DQN', type='values', question='q9')

    path_dueling = maze.get_path(q_network_dueling)
    maze.print_board(trajectories=[path_dueling], scenario='dueling_DQN', type='path', question='q9')

if __name__ == "__main__":
    run_problem()
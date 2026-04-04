import os

from matplotlib import pyplot as plt
import numpy as np

class biological_gene_network:
    GENE_TYPES = ['a1', 'a2', 'a3', 'a4']
    ACTIONS = {
        'a1': np.array([0, 0, 0, 0]),
        'a2':     np.array([0, 1, 0, 0]),
        'a3':    np.array([0, 0, 1, 0]),
        'a4':    np.array([0, 0, 0, 1])
    }
    C = np.array([
        [ 0,  0, -1,  0],
        [ 1,  0, -1, -1],
        [ 0,  1,  0,  0],
        [-1,  1,  1,  0]
    ])

    def __init__(self, p):
        self.p = p
        self.states = np.array([[int(b) for b in format(i, '04b')] for i in range(16)])
        self.transition_matrices = self.construct_transition_matrices(p)
        self.reward_matrices = self.construct_reward_matrices()

    def v(self, x):
        return (x > 0).astype(int)
    
    def construct_transition_matrices(self, p):
        # Transition probabilities for each action
        transition_matrices = {}
        for action_name, action_vector in self.ACTIONS.items():
            M = np.zeros((16, 16))
            for i in range(16):
                cs_a = self.v(self.C @ self.states[i]) ^ action_vector
                for j in range(16):
                    diff = np.sum(np.abs(self.states[j] - cs_a))
                    M[i, j] = p**diff * (1-p)**(4-diff)
            transition_matrices[action_name] = M
        return transition_matrices

    # R(s,a,s') = 5*sum(s') - ||a||
        # Expected reward vector R_a[i] = sum_j M(a)[i,j] * (5*sum(s_j) - ||a||)
    def construct_reward_matrices(self):
        rewards = {}
        gene_activation = np.array([5 * np.sum(s) for s in self.states])
        for action_name, action_vector in self.ACTIONS.items():
            action_cost = np.sum(action_vector)
            R_a = self.transition_matrices[action_name] @ gene_activation - action_cost
            rewards[action_name] = R_a
        return rewards
    
    def plot_avg_cumulative_rewards(self, all_rewards, labels, question='q2', scenario='avg_cumulative_rewards', type='avg_cumulative_rewards', T_max=100):
      fig, ax = plt.subplots(figsize=(10, 6))
      
      for rewards_across_runs, label in zip(all_rewards, labels):
          # rewards_across_runs is a list of 10 lists, each of length num_of_episodes
          avg_rewards = np.mean(np.array(rewards_across_runs), axis=0)
          ax.plot(range(len(avg_rewards)), avg_rewards, label=label)
      
      ax.set_xlabel('Episode number')
      ax.set_ylabel('Average Accumulated Reward')
      ax.set_title('Average Accumulated Reward vs Episode Number')
      ax.legend()
      plt.tight_layout()
      os.mkdir(question) if not os.path.exists(question) else None
      os.mkdir(question + '/' + scenario) if not os.path.exists(question + '/' + scenario) else None
      os.mkdir(question + '/' + scenario + '/' + type.split("_")[0]) if not os.path.exists(question + '/' + scenario + '/' + type.split("_")[0]) else None
      plt.savefig(question + '/' + scenario + '/' + type.split("_")[0] + '/' + type + '.png', dpi=150, bbox_inches='tight')
      plt.close()
    
    def q_learning_optimal_policy(self, γ, α, ϵ, num_of_episodes, T_max):
        action_list = list(self.ACTIONS.keys())
        Q_s_a = np.zeros((16, len(action_list)))
        optimal_policy = ['a1'] * 16
        episode_rewards = []

        for _ in range(num_of_episodes):
            s = np.random.randint(0, 16)
            total_reward = 0

            for _ in range(T_max):
                # ϵ-greedy action selection
                if np.random.rand() < ϵ:
                    action_idx = np.random.choice(len(action_list))
                else:
                    action_idx = np.argmax(Q_s_a[s])
                action = action_list[action_idx]

                # Sample next state using transition matrix
                next_s = np.random.choice(16, p=self.transition_matrices[action][s])
                reward = 5 * np.sum(self.states[next_s]) - np.sum(self.ACTIONS[action])

                best_next_q = np.max(Q_s_a[next_s])
                Q_s_a[s][action_idx] += α * (reward + γ * best_next_q - Q_s_a[s][action_idx])

                total_reward += reward
                s = next_s

            episode_rewards.append(total_reward)

        for s in range(16):
            optimal_policy[s] = action_list[np.argmax(Q_s_a[s])]

        return Q_s_a, optimal_policy, episode_rewards

    def SARSA_optimal_policy(self, γ, α, ϵ, num_of_episodes, T_max):
        action_list = list(self.ACTIONS.keys())
        Q_s_a = np.zeros((16, len(action_list)))
        optimal_policy = ['a1'] * 16
        episode_rewards = []

        for _ in range(num_of_episodes):
            s = np.random.randint(0, 16)
            total_reward = 0

            if np.random.rand() < ϵ:
                action_idx = np.random.choice(len(action_list))
            else:
                action_idx = np.argmax(Q_s_a[s])
            action = action_list[action_idx]

            for _ in range(T_max):
                next_s = np.random.choice(16, p=self.transition_matrices[action][s])
                reward = 5 * np.sum(self.states[next_s]) - np.sum(self.ACTIONS[action])

                if np.random.rand() < ϵ:
                    next_action_idx = np.random.choice(len(action_list))
                else:
                    next_action_idx = np.argmax(Q_s_a[next_s])
                next_action = action_list[next_action_idx]

                Q_s_a[s][action_idx] += α * (reward + γ * Q_s_a[next_s][next_action_idx] - Q_s_a[s][action_idx])

                total_reward += reward
                s = next_s
                action_idx = next_action_idx
                action = next_action

            episode_rewards.append(total_reward)

        for s in range(16):
            optimal_policy[s] = action_list[np.argmax(Q_s_a[s])]

        return Q_s_a, optimal_policy, episode_rewards

    def actor_critic_optimal_policy(self, γ, α, β, num_of_episodes, T_max):
        action_list = list(self.ACTIONS.keys())
        V_s = np.zeros(16)
        H_s_a = np.zeros((16, len(action_list)))
        optimal_policy = ['a1'] * 16
        episode_rewards = []

        for _ in range(num_of_episodes):
            s = np.random.randint(0, 16)
            total_reward = 0

            for _ in range(T_max):
                exp_values = np.exp(H_s_a[s])
                π = exp_values / np.sum(exp_values)
                action_idx = np.random.choice(len(action_list), p=π)
                action = action_list[action_idx]

                next_s = np.random.choice(16, p=self.transition_matrices[action][s])
                reward = 5 * np.sum(self.states[next_s]) - np.sum(self.ACTIONS[action])

                td_error = reward + γ * V_s[next_s] - V_s[s]
                V_s[s] += α * td_error
                H_s_a[s][action_idx] += β * td_error * (1 - π[action_idx])

                total_reward += reward
                s = next_s

            episode_rewards.append(total_reward)

        for s in range(16):
            exp_values = np.exp(H_s_a[s])
            π = exp_values / np.sum(exp_values)
            optimal_policy[s] = action_list[np.argmax(π)]

        return V_s, optimal_policy, episode_rewards
    
    def SARSA_lambda_optimal_policy(self, γ, α, ϵ, λ, num_of_episodes, T_max):
        # Initialize Q(s, a) arbitrarily and e(s, a) = 0, for all s, a
        action_list = list(self.ACTIONS.keys())
        Q_s_a = np.zeros((16, len(action_list)))
        optimal_policy = ['a1'] * 16
        episode_rewards = []

        # Iterate until s is terminal 
        for _ in range(num_of_episodes):
            # Initialize s, a
            s = np.random.randint(0, 16)
            total_reward = 0
            # e(s, a) = 0, for all s, a
            E = np.zeros((16, len(action_list)))  # eligibility traces, reset each episode
            
            if np.random.rand() < ϵ:
                action_idx = np.random.choice(len(action_list))
            else:
                action_idx = np.argmax(Q_s_a[s])
            action = action_list[action_idx]

            for _ in range(T_max):
                # Take action a, observe r, s′
                next_s = np.random.choice(16, p=self.transition_matrices[action][s])
                reward = 5 * np.sum(self.states[next_s]) - np.sum(self.ACTIONS[action])

                # Choose a′ from s′ using policy derived from Q (e.g., ε-greedy)
                if np.random.rand() < ϵ:
                    next_action_idx = np.random.choice(len(action_list))
                else:
                    next_action_idx = np.argmax(Q_s_a[next_s])
                next_action = action_list[next_action_idx]

                # δ ← r + γQ(s′, a′) − Q(s, a)
                td_error = reward + γ * Q_s_a[next_s][next_action_idx] - Q_s_a[s][action_idx]

                # e(s, a) ← e(s, a) + 1
                E[s][action_idx] += 1

                # Q(s, a) ← Q(s, a) + α δ e(s, a)
                Q_s_a += α * td_error * E

                # e(s, a) ← γ λ e(s, a)
                E *= γ * λ

                total_reward += reward
                # s ← s′
                s = next_s

                # a ← a′
                action_idx = next_action_idx
                action = next_action

            episode_rewards.append(total_reward)

        for s in range(16):
            optimal_policy[s] = action_list[np.argmax(Q_s_a[s])]

        return Q_s_a, optimal_policy, episode_rewards
    
    def execute_greedy_policy(self, policy, n_episodes=100, T_max=100):
        visitation_counts = np.zeros(16)
        for _ in range(n_episodes):
            s = np.random.randint(0, 16)
            for _ in range(T_max):
                action = policy[s]
                s = np.random.choice(16, p=self.transition_matrices[action][s])
                visitation_counts[s] += 1
        return visitation_counts

    def plot_visitation_counts(self, all_counts, labels, question='problem2', scenario='visitation', type='visitation'):
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(16)
        width = 0.2
        for idx, (counts, label) in enumerate(zip(all_counts, labels)):
            ax.bar(x + idx * width, counts, width, label=label)
        ax.set_xlabel('State')
        ax.set_ylabel('Visitation Count')
        ax.set_title('State Visitation Counts (100 episodes)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(s) for s in range(16)])
        ax.legend()
        plt.tight_layout()
        os.mkdir(question) if not os.path.exists(question) else None
        os.mkdir(question + '/' + scenario) if not os.path.exists(question + '/' + scenario) else None
        plt.savefig(question + '/' + scenario + '/' + type + '.png', dpi=150, bbox_inches='tight')
        plt.close()
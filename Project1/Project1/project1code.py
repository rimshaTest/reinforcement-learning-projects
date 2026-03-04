# -*- coding: utf-8 -*-
"""
Project 1  
EECE 5614  
StudentName: Rimsha Kayastha
"""

import numpy as np
import random
import matplotlib.pyplot as plt

class UtilityFunctions():

  def get_reward_a(self):
    μ = 8
    σ = np.sqrt(20)
    return np.random.normal(μ, σ)

  def get_reward_b(self):
    μ_1 = 8
    σ_1 = 15
    μ_2 = 14
    σ_2 = 10
    return np.random.normal(μ_1, σ_1) if np.random.random() < 0.5 else np.random.normal(μ_2, σ_2)

  def get_average_acc_reward(self, rewards):
    return np.mean(rewards)

class EpsilonGreedy(UtilityFunctions):

  def __init__(self): 
    self.part1_learning_rates = [
        lambda k: 1,
        lambda k: 0.9**k,
        lambda k: 1/(1+np.log(1+k)),
        lambda k: 1/k
    ]
    self.part1_epsilon_policies = [0, 0.1, 0.2, 0.5]
    self.part1_combinations = [(a,b) for a in self.part1_learning_rates for b in self.part1_epsilon_policies]
    self.part1_learning_rates_str = ['1', '0.9**k', '1/(1+np.log(1+k))', '1/k']
    self.part1_combinations_str = [(a,b) for a in self.part1_learning_rates_str for b in self.part1_epsilon_policies]
    self.part1_results = None

    self.part2_α = lambda k: 0.1
    self.part2_ϵ = 0.1
    self.part2_combinations = [(self.part2_α, self.part2_ϵ)]
    self.part2_Q_a1_a2_values = [(0,0), (8,11), (20,20)]
    self.part2_results = None

  def select_greedy_epsilon_action(self, Q_a, Q_b, ϵ):
    if random.random() < ϵ:
      return random.choice(['a', 'b']) # exploration
    else:
      return 'a' if Q_a > Q_b else 'b' # exploitation

  def rl_bandit_100loops(self, combinations, a, b):
    average_Q_a_values = []
    average_Q_b_values = []
    average_acc_rewards = []
    for alpha, ϵ in combinations:
      Q_a_values = []
      Q_b_values = []
      all_runs_acc_rewards = []

      for j in range(100):
        rewards = []
        acc_rewards = []
        Q_a, Q_b = a, b
        for i in range(1000):
          α = alpha(i+1)
          action = self.select_greedy_epsilon_action(Q_a, Q_b, ϵ)

          if action == 'a':
            r_a = self.get_reward_a()
            Q_a = Q_a + α * (r_a - Q_a)
            rewards.append(r_a)
          else:
            r_b = self.get_reward_b()
            Q_b = Q_b + α * (r_b - Q_b)
            rewards.append(r_b)
          acc_rewards.append(self.get_average_acc_reward(rewards))

        Q_a_values.append(Q_a)
        Q_b_values.append(Q_b)
        all_runs_acc_rewards.append(acc_rewards)
      average_acc_rewards.append(np.mean(all_runs_acc_rewards, axis=0))

      average_Q_a_values.append(np.mean(Q_a_values))
      average_Q_b_values.append(np.mean(Q_b_values))
    return average_Q_a_values, average_Q_b_values, average_acc_rewards
  
  """Part 1"""
  def part1(self):
    average_Q_a_values, average_Q_b_values, average_acc_rewards_epsilon = self.rl_bandit_100loops(self.part1_combinations, 0, 0)

    epsilon_greedy_results = {}
    for rewards, lr in zip(average_acc_rewards_epsilon, self.part1_combinations_str):
      epsilon_greedy_results[lr] = rewards
    self.part1_results = epsilon_greedy_results

    # Plot - 4 subplots (one per learning rate)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['blue', 'orange', 'green', 'red']

    for α_idx in range(4):  # 4 learning rates
        ax = axes[α_idx]

        for ε_idx, (ε, color) in enumerate(zip(self.part1_epsilon_policies, colors)):
            # Index into average_acc_rewards
            combo_idx = α_idx * 4 + ε_idx
            avg_acc_rewards = average_acc_rewards_epsilon[combo_idx]

            ax.plot(range(1, len(avg_acc_rewards)+1), avg_acc_rewards,
                    label=f'ε={ε}', color=color, linewidth=1.5)

        ax.set_xlabel('Time(t)', fontsize=12)
        ax.set_ylabel('Average Accumulated Reward', fontsize=12)
        ax.set_title(f'{self.part1_learning_rates_str[α_idx]}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/part1_results.png', dpi=300, bbox_inches='tight')
    plt.close()

  """Part 2"""
  def part2(self):
    all_results = []
    for (i, j) in self.part2_Q_a1_a2_values:
      average_Q_a1_values_0, average_Q_a2_values_0, average_acc_rewards = self.rl_bandit_100loops(self.part2_combinations, i, j)
      all_results.append({'label': f'Q_a1 = {i}, Q_a2 = {j}', 'data': average_acc_rewards[0]})
    self.part2_results = all_results

    plt.figure(figsize=(10, 6))

    colors = ['blue', 'orange', 'green']

    for result, color in zip(all_results, colors):
        plt.plot(range(1, 1001), result['data'],
                label=result['label'], color=color, linewidth=2)

    plt.xlabel('Time(t)', fontsize=12)
    plt.ylabel('Average Accumulated Reward', fontsize=12)
    plt.title('Epsilon-greedy with different Initial Q-values (α=0.1, ε=0.1)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/part2_results.png', dpi=300, bbox_inches='tight')
    plt.close()

class GradientBandit(UtilityFunctions):

  def __init__(self):
    self.α = 0.1
    self.part3_results = None

  def calculate_policy_preferences(self, expected_H_a, expected_H_b):
    return np.exp(expected_H_a)/(np.exp(expected_H_a)+np.exp(expected_H_b))

  def select_action(self, π_a1, π_a2):
    return 'a1' if np.random.rand()< π_a1 else 'a2'

  def rl_gradient_bandit_100loops(self, α, H1_a1, H1_a2):
      H_a1_values = []
      H_a2_values = []
      all_runs_acc_rewards = []

      for j in range(100):
        rewards = []
        acc_rewards = []
        H_a1, H_a2 = H1_a1, H1_a2
        for i in range(1000):
          π_a1 = self.calculate_policy_preferences(H_a1, H_a2)
          π_a2 = self.calculate_policy_preferences(H_a2, H_a1)
          action = self.select_action(π_a1, π_a2)

          if action == 'a1':
            r = self.get_reward_a()
            rewards.append(r)
            acc_r = self.get_average_acc_reward(rewards)
            H_a1 = H_a1 + α * (r - acc_r) * (1 - π_a1)
            H_a2 = H_a2 - α * (r - acc_r) * π_a2
          else:
            r = self.get_reward_b()
            rewards.append(r)
            acc_r = self.get_average_acc_reward(rewards)
            H_a2 = H_a2 + α * (r - acc_r) * (1 - π_a2)
            H_a1 = H_a1 - α * (r - acc_r) * π_a1
          acc_rewards.append(acc_r)

        H_a1_values.append(H_a1)
        H_a2_values.append(H_a2)
        all_runs_acc_rewards.append(acc_rewards)

      return np.mean(H_a1_values), np.mean(H_a2_values), np.mean(all_runs_acc_rewards, axis=0)
  
  """Part 3"""
  def part3(self, part2):
    average_H_a1_values, average_H_a2_values, average_acc_rewards_gradient = self.rl_gradient_bandit_100loops(self.α, 0, 0)
    self.part3_results = average_acc_rewards_gradient

    plt.figure(figsize=(10, 6))

    # Plot Gradient Bandit
    plt.plot(range(1, 1001), average_acc_rewards_gradient,
            label='Gradient Bandit (α=0.1, H=[0,0])',
            color='blue', linewidth=2)

    # Plot ε-greedy
    plt.plot(range(1, 1001), part2[0]['data'],
            label='ε-greedy (α=0.1, ε=0.1, Q=[0,0])',
            color='orange', linewidth=2)

    plt.xlabel('Time(t)', fontsize=12)
    plt.ylabel('Average Accumulated Reward', fontsize=12)
    plt.title('Gradient Bandit vs ε-greedy', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/part3_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


class UCB(UtilityFunctions):

  def __init__(self):
    self.c_values = [2, 5, 100]
    self.part4_results = None

  def select_action_UCB(self, Q_a1, Q_a2, N_a1, N_a2, i, c):
    if N_a1 == 0:
        return 'a1'
    if N_a2 == 0:
        return 'a2'

    # UCB formula: Q + c * sqrt(ln(k) / N)
    ucb_a1 = Q_a1 + c * np.sqrt(np.log(i) / N_a1)
    ucb_a2 = Q_a2 + c * np.sqrt(np.log(i) / N_a2)

    return 'a1' if ucb_a1 > ucb_a2 else 'a2'

  def rl_bandit_100loops_UCB(self, c):
    Q_a1_values, Q_a2_values = [], []
    all_runs_acc_rewards = []

    for j in range(100):
      rewards = []
      acc_rewards = []
      Q_a1, Q_a2 = 0, 0
      N_a1, N_a2 = 0, 0

      for i in range(1000):
        action = self.select_action_UCB(Q_a1, Q_a2, N_a1, N_a2, i, c)

        if action == 'a1':
          N_a1+=1
          r_a1 = self.get_reward_a()
          Q_a1 = Q_a1 + (1/N_a1) * (r_a1 - Q_a1)
          rewards.append(r_a1)
        else:
          N_a2+=1
          r_a2 = self.get_reward_b()
          Q_a2 = Q_a2 + (1/N_a2) * (r_a2 - Q_a2)
          rewards.append(r_a2)

        acc_rewards.append(self.get_average_acc_reward(rewards))

      Q_a1_values.append(Q_a1)
      Q_a2_values.append(Q_a2)
      all_runs_acc_rewards.append(acc_rewards)

    average_acc_rewards=np.mean(all_runs_acc_rewards, axis=0)
    average_Q_a1_values= np.mean(Q_a1_values)
    average_Q_a2_values =np.mean(Q_a2_values)

    return average_Q_a1_values, average_Q_a2_values, average_acc_rewards

  """Part 4"""
  def part4(self, epsilon_greedy_results, average_acc_rewards_gradient):
    c_values_UCB_result = {}
    for c in self.c_values:
      average_Q_a1_values_UCB, average_Q_a2_values_UCB, average_acc_rewards_UCB = self.rl_bandit_100loops_UCB(c)
      c_values_UCB_result[c] = average_acc_rewards_UCB
    
    self.part4_results = c_values_UCB_result

    plt.figure(figsize=(10, 6))

    colors = ['blue', 'orange', 'green']
    for c, color in zip(self.c_values, colors):
        plt.plot(range(1, 1001), c_values_UCB_result[c],
                label=f'UCB c={c}', color=color, linewidth=2)

    plt.xlabel('Time(t)', fontsize=12)
    plt.ylabel('Average Accumulated Reward', fontsize=12)
    plt.title('UCB Performance for Different c Values', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/part3_UCB_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    comparable_results = {
        'UCB c=5': c_values_UCB_result[5],
        'ε-greedy α = 0.9^k ϵ=0.5': epsilon_greedy_results[('1/k', 0)],
        'gradient-bandit': average_acc_rewards_gradient
    }

    plt.figure(figsize=(10, 6))

    colors = ['blue', 'orange', 'green']
    for strategy, color in zip(comparable_results.keys(), colors):
        plt.plot(range(1, 1001), comparable_results[strategy],
                label=f'{strategy}', color=color, linewidth=2)

    plt.xlabel('Time(t)', fontsize=12)
    plt.ylabel('Average Accumulated Reward', fontsize=12)
    plt.title('Performance for different strategies', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/part4_UCB_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
  # Epsilon-greedy
  epsilon_greedy = EpsilonGreedy()

  ## Part 1
  epsilon_greedy.part1()
  print("Part 1 completed")

  ## Part 2
  epsilon_greedy.part2()
  print("Part 2 completed")

  # Gradient Bandit
  gradient_bandit = GradientBandit()

  ## Part 3
  gradient_bandit.part3(epsilon_greedy.part2_results)
  print("Part 3 completed")

  # UCB
  ucb = UCB()

  ## part 4
  ucb.part4(epsilon_greedy.part1_results, gradient_bandit.part3_results)
  print("Part 4 completed")



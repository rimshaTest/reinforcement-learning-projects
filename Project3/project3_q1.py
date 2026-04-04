# -*- coding: utf-8 -*-
import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

class Maze():
    actions = {
        '←': (0, -1),
        '→': (0, 1),
        '↑': (-1, 0),
        '↓': (1, 0)
    }
    
    def __init__(self, length, width, walls, oils, bumps, goal, start, penalties):
      self.length = length
      self.width = width
      self.goal = goal
      self.start = start
      self.penalties = penalties
      self.actions = Maze.actions
      self.board = [[' ' for _ in range(width)] for _ in range(length)]
      self.rewards = [[-1 for _ in range(width)] for _ in range(length)]
      self.values = [[0 for _ in range(width)] for _ in range(length)]
      self.policies = [[' ' for _ in range(width)] for _ in range(length)]
      self.colors = ['black', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange']
      self.initialize_board(walls, oils, bumps, goal, start)
      self.print_board()
      self.initialize_rewards(walls, oils, bumps, goal, penalties)

    def initialize_board(self, walls, oils, bumps, goal, start):
      for wall in walls:
        self.board[wall[0]][wall[1]] = 'W'

      for oil in oils:
        self.board[oil[0]][oil[1]] = 'O'

      for bump in bumps:
        self.board[bump[0]][bump[1]] = 'B'

      self.board[goal[0]][goal[1]] = 'G'
      self.board[start[0]][start[1]] = 'S'

    def initialize_policy(self, action = '←'):
      for i in range(self.length):
        for j in range(self.width):
          if self.board[i][j] == 'G':
            self.policies[i][j] = 'G'
          if self.board[i][j] not in ['W', 'G']:
            self.policies[i][j] = action
            
    def initialize_rewards(self, walls, oils, bumps, goal, penalties):
      for wall in walls:
        self.rewards[wall[0]][wall[1]] += penalties['wall']

      for oil in oils:
        self.rewards[oil[0]][oil[1]] += penalties['oil']

      for bump in bumps:
        self.rewards[bump[0]][bump[1]] += penalties['bump']

      self.rewards[goal[0]][goal[1]] += penalties['goal']

    def print_board(self, boards=[], scenario='', type='maze', question='general', trajectories=[]):
      fig, ax = plt.subplots(figsize=(14, 14))
      
      # Draw maze background
      for i in range(self.length):
          for j in range(self.width):
              cell = self.board[i][j]
              if cell == 'W':
                  color = 'black'
              elif cell == 'G':
                  color = 'lightgreen'
              elif cell == 'S':
                  color = 'lightblue'
              elif cell == 'O':
                  color = 'red'
              elif cell == 'B':
                  color = 'orange'
              else:
                  color = 'white'
              ax.add_patch(patches.Rectangle((j, self.length-1-i), 1, 1,
                                            facecolor=color, edgecolor='gray'))
              
              # Draw board content (values/policies)
              for iteration, board in enumerate(boards):
                  value = board[i][j]
                  if value not in [' ', '']:
                      if isinstance(value, float):
                          display = str(round(value, 2))
                          font = 10
                      else:
                          display = str(value)
                          font = 20
                      ax.text(j + 0.5, self.length - 0.5 - i, display,
                              ha='center', va='center', fontsize=font, color=self.colors[iteration])
      
      # Draw trajectories as lines
      for idx, traj in enumerate(trajectories):
          xs = [j + 0.5 for (i, j) in traj]
          ys = [self.length - 0.5 - i for (i, j) in traj]
          ax.plot(xs, ys, color=self.colors[idx], linewidth=2, 
                  marker='o', markersize=10, label=f'Trajectory {idx+1}')
      
      if trajectories:
          ax.legend()

      ax.set_xlim(0, self.width)
      ax.set_ylim(0, self.length)
      ax.set_aspect('equal')
      ax.axis('off')
      plt.tight_layout()
      os.mkdir(question) if not os.path.exists(question) else None
      os.mkdir(question + '/' + scenario) if not os.path.exists(question + '/' + scenario) else None
      os.mkdir(question + '/' + scenario + '/' + type.split("_")[0]) if not os.path.exists(question + '/' + scenario + '/' + type.split("_")[0]) else None
      plt.savefig(question + '/' + scenario + '/' + type.split("_")[0] + "/"  + type + '.png', dpi=150, bbox_inches='tight')
      plt.close()

    def plot_avg_cumulative_rewards(self, all_rewards, labels, question='q1', scenario='avg_cumulative_rewards', type='avg_cumulative_rewards', T_max=400):
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

    def get_optimal_policy(self, policy_board):
      current_cell = self.start
      path = [['  ' for _ in range(self.width)] for _ in range(self.length)]
      while current_cell != self.goal:
        action = policy_board[current_cell[0]][current_cell[1]]
        if action == 'G':
          break
        dx, dy = self.actions[action]
        next_cell = (current_cell[0] + dx, current_cell[1] + dy)
        if (next_cell[0] < 0 or next_cell[0] >= self.length or
            next_cell[1] < 0 or next_cell[1] >= self.width or
            self.board[next_cell[0]][next_cell[1]] == 'W'):
          break
        path[current_cell[0]][current_cell[1]] = action
        current_cell = next_cell
      
      return path

    def sample_trajectory(self, policy, p, T_max=400):
      current = self.start
      trajectory = [(current[0], current[1])]
      total_reward = 0
      cumulative_rewards = [0]
      
      for _ in range(T_max):
          if current == self.goal:
              break
          
          action = policy[current[0]][current[1]]
          next_states = self.get_next_states(current[0], current[1], action, p)
          
          probs = [prob for prob, _ in next_states]
          states = [state for _, state in next_states]
          idx = np.random.choice(len(states), p=probs)
          next_state = states[idx]
          
          total_reward += self.rewards[next_state[0]][next_state[1]]
          cumulative_rewards.append(total_reward)
          trajectory.append(next_state)
          current = next_state
      
      return trajectory, cumulative_rewards

    def get_next_states(self, i, j, a, p):
      dx, dy = self.actions[a]
      results = []
      
      for action, (ddx, ddy) in self.actions.items():
          # Skip opposite direction
          if ddx == -dx and ddy == -dy:
              continue
          
          # Intended action gets 1-p, perpendiculars get p/2
          if ddx == dx and ddy == dy:
              prob = 1 - p
          else:
              prob = p / 2
          
          ni, nj = i + ddx, j + ddy
          
          # Out of bounds or wall → stay
          if (ni < 0 or ni >= self.length or
              nj < 0 or nj >= self.width or
              self.board[ni][nj] == 'W'):
              ni, nj = i, j
          
          results.append((prob, (ni, nj)))
      
      return results
    
    def greedy_policy_finds_goal(self, Q_s_a, action_list):
      current = self.start
      visited = set()
      while current != self.goal:
          if current in visited:
              return False  # stuck in a loop
          visited.add(current)
          i, j = current
          action_idx = np.argmax(Q_s_a[i][j])
          action = action_list[action_idx]
          di, dj = self.actions[action]
          ni, nj = i + di, j + dj
          if (ni < 0 or ni >= self.length or
              nj < 0 or nj >= self.width or
              self.board[ni][nj] == 'W'):
              ni, nj = i, j
          if (ni, nj) == current:
              return False  # hitting a wall and staying, stuck
          current = (ni, nj)
      return True
    
    def greedy_policy_finds_goal_ac(self, H_s_a, action_list):
      current = self.start
      visited = set()
      while current != self.goal:
          if current in visited:
              return False
          visited.add(current)
          i, j = current
          exp_values = np.exp(H_s_a[i][j])
          π = exp_values / np.sum(exp_values)
          action = action_list[np.argmax(π)]
          di, dj = self.actions[action]
          ni, nj = i + di, j + dj
          if (ni < 0 or ni >= self.length or
              nj < 0 or nj >= self.width or
              self.board[ni][nj] == 'W'):
              ni, nj = i, j
          if (ni, nj) == current:
              return False
          current = (ni, nj)
      return True

    def q_learning_optimal_policy(self, γ, α, ϵ, num_of_episodes, T_max):
      # Initialize Q(s, a) arbitrarily
      Q_s_a = np.zeros((self.length, self.width, len(self.actions)))
      action_list = list(self.actions.keys())
      optimal_policy = np.full((self.length, self.width), '←')
      episode_rewards = []

      first_successful_episode = None

      for e in range(num_of_episodes):
        # Initialize s
        current = self.start
        total_reward = 0  # reset per episode

        for _ in range(T_max):
          # Iterate until until s is terminal 
          if current == self.goal:
              break

          i, j = current

          # Choose a from s using policy derived from Q (ϵ-greedy)
          if np.random.rand() < ϵ:
              action_idx = np.random.choice(len(action_list))
          else:
              action_idx = np.argmax(Q_s_a[i][j])
          action = action_list[action_idx]

          # Take action a, observe r, s′
          di, dj = self.actions[action]
          ni, nj = i + di, j + dj
          if (ni < 0 or ni >= self.length or
              nj < 0 or nj >= self.width or
              self.board[ni][nj] == 'W'):
              ni, nj = i, j
          next_state = (ni, nj)

          # Q(s, a) ← Q(s, a) + α [r + γ max_a′ Q(s′, a′) − Q(s, a)]
          reward = self.rewards[next_state[0]][next_state[1]]
          best_next_q = 0 if next_state == self.goal else np.max(Q_s_a[next_state[0]][next_state[1]])
          Q_s_a[i][j][action_idx] += α * (reward + γ * best_next_q - Q_s_a[i][j][action_idx])

          total_reward += reward # accumulate reward for this episode
          # s ← s′
          current = next_state
        episode_rewards.append(total_reward)

        # Checking if greedy policy finds goal for the first time
        if first_successful_episode is None:
          if self.greedy_policy_finds_goal(Q_s_a, action_list):
              first_successful_episode = e + 1

      # Extract optimal policy from Q-values
      for i in range(self.length):
        for j in range(self.width):
          if self.board[i][j] not in ['W', 'G']:
            best_action_idx = np.argmax(Q_s_a[i][j])
            optimal_policy[i][j] = action_list[best_action_idx]

      print(f"First successful episode for Q-Learning: {first_successful_episode if first_successful_episode is not None else 1000}")

      return Q_s_a, optimal_policy, episode_rewards
    
    def SARSA_optimal_policy(self, γ, α, ϵ, num_of_episodes, T_max):
      # Initialize Q(s, a) arbitrarily
      Q_s_a = np.zeros((self.length, self.width, len(self.actions)))
      action_list = list(self.actions.keys())
      optimal_policy = np.full((self.length, self.width), '←')
      episode_rewards = []

      first_successful_episode = None

      for e in range(num_of_episodes):
        # Initialize s
        current = self.start
        i, j = current
        total_reward = 0  # reset per episode


        # Choose a from s using policy derived from Q (e.g., ϵ-greedy)
        if np.random.rand() < ϵ:
            action_idx = np.random.choice(len(action_list))
        else:
            action_idx = np.argmax(Q_s_a[i][j])
        action = action_list[action_idx]

        for _ in range(T_max):
          # Iterate until s is terminal 
          if current == self.goal:
              break

          # Take action a, observe r, s′
          di, dj = self.actions[action]
          ni, nj = i + di, j + dj
          if (ni < 0 or ni >= self.length or
              nj < 0 or nj >= self.width or
              self.board[ni][nj] == 'W'):
              ni, nj = i, j
          next_state = (ni, nj)

          reward = self.rewards[next_state[0]][next_state[1]]
          
          # Choose a′ from s′ using policy derived from Q (ϵ-greedy)
          if next_state == self.goal:
              next_action_idx = 0  # doesn't matter, next_q will be 0
              next_q = 0
          else:
              if np.random.rand() < ϵ:
                  next_action_idx = np.random.choice(len(action_list))
              else:
                  next_action_idx = np.argmax(Q_s_a[next_state[0]][next_state[1]])
              next_q = Q_s_a[next_state[0]][next_state[1]][next_action_idx]

          # Q(s, a) ← Q(s, a) + α [r + γ Q(s′, a′) − Q(s, a)]
          Q_s_a[i][j][action_idx] += α * (reward + γ * next_q - Q_s_a[i][j][action_idx])

          total_reward += reward # accumulate reward for this episode

          # s ← s′
          current = next_state
          i, j = current

          # a ← a′
          action_idx = next_action_idx
          action = action_list[action_idx]
        episode_rewards.append(total_reward)

        # Checking if greedy policy finds goal for the first time
        if first_successful_episode is None:
          if self.greedy_policy_finds_goal(Q_s_a, action_list):
              first_successful_episode = e + 1

      # Extract optimal policy from Q-values
      for i in range(self.length):
        for j in range(self.width):
          if self.board[i][j] not in ['W', 'G']:
            best_action_idx = np.argmax(Q_s_a[i][j])
            optimal_policy[i][j] = action_list[best_action_idx]
      
      print(f"First successful episode for SARSA: {first_successful_episode if first_successful_episode is not None else 1000}")

      return Q_s_a, optimal_policy, episode_rewards
    
    def actor_critic_optimal_policy(self, γ, α, β, num_of_episodes, T_max):
      # InitializeV (s) = 0, H(s, a) = 0, for all s ∈ S, a ∈ A
      V_s = np.zeros((self.length, self.width))
      H_s_a = np.zeros((self.length, self.width, len(self.actions)))
      action_list = list(self.actions.keys())
      optimal_policy = np.full((self.length, self.width), '←')
      episode_rewards = []
      
      first_successful_episode = None
      for e in range(num_of_episodes):
        # Start from a random state s0 ∈ S, t = 0
        current = self.start
        total_reward = 0  # reset per episode

        for _ in range(T_max):
          # Iterate until s_t is terminal
          if current == self.goal:
              break
          i, j = current

          # Select action: a_t ∼ π(· | s_t) where π(a | s) = e^H(s,a)/(∑_{a′∈A} e^H(s,a′))
          exp_values = np.exp([H_s_a[i][j][k] for k in range(len(action_list))])
          π = exp_values / np.sum(exp_values)
          action_idx = np.random.choice(len(action_list), p=π)
          action = action_list[action_idx]

          # Take action a_t, move to state s_t+1 and observe R_t+1
          di, dj = self.actions[action]
          ni, nj = i + di, j + dj
          if (ni < 0 or ni >= self.length or
              nj < 0 or nj >= self.width or
              self.board[ni][nj] == 'W'):
              ni, nj = i, j
          next_state = (ni, nj)

          reward = self.rewards[next_state[0]][next_state[1]]
          # Calculate TD Error:δ_t = R_t+1 + γ V (s_t+1) − V (s_t)
          next_v = 0 if next_state == self.goal else V_s[next_state[0]][next_state[1]]
          td_error = reward + γ * next_v - V_s[i][j]
          
          # Update value function: V(s_t) = V (s_t) + α δ_t
          V_s[i][j] += α * td_error

          # H(s_t, a_t) = H(s_t, a_t) + β δ_t (1 − π(a_t | s_t))
          H_s_a[i][j][action_idx] += β * td_error * (1 - π[action_idx])

          total_reward += reward # accumulate reward for this episode

          current = next_state

        episode_rewards.append(total_reward)

        # Checking if greedy policy finds goal for the first time
        if first_successful_episode is None:
          if self.greedy_policy_finds_goal(H_s_a, action_list):
              first_successful_episode = e + 1

      # Extract optimal policy from learned policy
      for i in range(self.length):
        for j in range(self.width):
            if self.board[i][j] not in ['W', 'G']:
                exp_values = np.exp([H_s_a[i][j][k] for k in range(len(action_list))])
                π = exp_values / np.sum(exp_values)
                optimal_policy[i][j] = action_list[np.argmax(π)]
      
      print(f"First successful episode for Actor-Critic: {first_successful_episode if first_successful_episode is not None else 1000}")

      return V_s, optimal_policy, episode_rewards
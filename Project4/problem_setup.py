# -*- coding: utf-8 -*-
import os
from random import random

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import torch

class Maze():
    actions = {
        '←': (0, -1),
        '→': (0, 1),
        '↑': (-1, 0),
        '↓': (1, 0)
    }
    
    def __init__(self, length, width, walls, yellows, reds, goal, start, penalties):
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
      self.initialize_board(walls, yellows, reds, goal, start)
      self.print_board()
      self.initialize_rewards(walls, yellows, reds, goal, penalties)
      self.print_board(boards=[self.rewards], type='rewards')

    def initialize_board(self, walls, yellows, reds, goal, start):
      for wall in walls:
        self.board[wall[0]][wall[1]] = 'W'

      for yellow in yellows:
        self.board[yellow[0]][yellow[1]] = 'Y'

      for red in reds:
        self.board[red[0]][red[1]] = 'R'

      self.board[goal[0]][goal[1]] = 'G'
      self.board[start[0]][start[1]] = 'S'

    def initialize_policy(self, action = '←'):
      for i in range(self.length):
        for j in range(self.width):
          if self.board[i][j] == 'G':
            self.policies[i][j] = 'G'
          if self.board[i][j] not in ['W', 'G']:
            self.policies[i][j] = action
            
    def initialize_rewards(self, walls, yellows, reds, goal, penalties):
      for wall in walls:
        self.rewards[wall[0]][wall[1]] += penalties['wall']

      for yellow in yellows:
        self.rewards[yellow[0]][yellow[1]] += penalties['yellow']

      for red in reds:
        self.rewards[red[0]][red[1]] += penalties['red']

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
                  color = 'blue'
              elif cell == 'Y':
                  color = 'yellow'
              elif cell == 'R':
                  color = 'red'
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
                          font = 20
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

    def step(self, state, action_idx, p=0.025):
        i, j = state
        action = list(self.actions.keys())[action_idx]
        next_states = self.get_next_states(i, j, action, p)
        probs = [prob for prob, _ in next_states]
        states = [s for _, s in next_states]
        idx = np.random.choice(len(states), p=probs)
        next_state = states[idx]
        reward = self.rewards[next_state[0]][next_state[1]]
        done = (next_state == self.goal)
        return next_state, reward, done

    def random_initial_state(self):
        while True:
            i = np.random.randint(0, self.length)
            j = np.random.randint(0, self.width)
            if self.board[i][j] not in ['W', 'G']:
                return (i, j)
            
    def get_policy(self, q_network):
      action_symbols = ['←', '→', '↑', '↓']
      policy = [[' ' for _ in range(self.width)] for _ in range(self.length)]
      
      for i in range(self.length):
          for j in range(self.width):
              if self.board[i][j] == 'W':
                  continue
              elif self.board[i][j] == 'G':
                  policy[i][j] = 'G'
              else:
                  state_tensor = torch.FloatTensor([i, j])
                  with torch.no_grad():
                      q_values = q_network(state_tensor)
                  policy[i][j] = action_symbols[q_values.argmax().item()]
      
      return policy
    
    def get_state_values(self, q_network):
      values = [[' ' for _ in range(self.width)] for _ in range(self.length)]
      
      for i in range(self.length):
          for j in range(self.width):
              if self.board[i][j] == 'W':
                  continue
              elif self.board[i][j] == 'G':
                  values[i][j] = 100.0
              else:
                  state_tensor = torch.FloatTensor([i, j])
                  with torch.no_grad():
                      q_values = q_network(state_tensor)
                  values[i][j] = round(q_values.max().item(), 2)
      
      return values
    
    def get_path(self, q_network, p=0.025):
      action_symbols = ['←', '→', '↑', '↓']
      current = self.start
      path = [current]
      
      for _ in range(50):  # T_epi = 50
          if current == self.goal:
              break
          
          state_tensor = torch.FloatTensor([current[0], current[1]])
          with torch.no_grad():
              q_values = q_network(state_tensor)
          action_idx = q_values.argmax().item()
          
          next_state, reward, done = self.step(current, action_idx, p)
          path.append(next_state)
          current = next_state
          
          if done:
              break
      
      return path
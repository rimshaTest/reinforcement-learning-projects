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
      plt.savefig(question + '/' + scenario + ('_' if scenario else '') + type + '.png', dpi=150, bbox_inches='tight')
      plt.close()

    def plot_avg_cumulative_rewards(self, p_values, all_rewards, question='q1_3b', scenario='avg_cumulative_rewards', T_max=400):
      fig, ax = plt.subplots(figsize=(10, 6))
      
      for rewards, p in zip(all_rewards, p_values):
          # Plotting horizon Tp = max termination time
          T_p = min(max(len(r) for r in rewards) - 1, T_max)
          
          # Pad shorter trajectories with their final value
          padded = []
          for r in rewards:
              padded_r = r + [r[-1]] * (T_p - len(r) + 1)
              padded.append(padded_r)
          
          # Compute average cumulative reward G_bar(t)
          G_bar = [np.mean([padded[i][t] for i in range(len(rewards))]) for t in range(T_p + 1)]
          
          ax.plot(range(T_p + 1), G_bar, label=f'p={p}')
      
      ax.set_xlabel('Time step t')
      ax.set_ylabel('Average Cumulative Reward')
      ax.set_title('Average Cumulative Reward Curves')
      ax.legend()
      plt.tight_layout()
      os.mkdir(question) if not os.path.exists(question) else None
      plt.savefig(question + '/' + scenario + '.png', dpi=150, bbox_inches='tight')
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

    def policy_iteration_optimal_policy(self, p, γ, θ ):
      # Initialization
      ## Maximum allowed iterations
      T_max = 400
      ## Choosing initial policy with all left
      self.initialize_policy( '←' )
      optimal_policy = [row[:] for row in self.policies]
      ## Initializing V_s
      V_s = np.zeros((self.length, self.width))
      # Iteration count
      count = 0

      ## Loop until policy is stable or until loop exceeds T_max
      while True:
        count += 1

        # Policy Evaluation
        ## Iterate through policies to find optimal policy
        while True:
          V_diff = 0
          ## Iterate through all states
          for i in range(self.length):
            for j in range(self.width):
              if self.board[i][j] in ['W', 'G']:
                continue
              ## Get the v, policy, reward, and probability of each state
              v = V_s[i][j]
              a = optimal_policy[i][j]

              ## Bellman's equation
              total = 0
              for prob, (x, y) in self.get_next_states(i, j, a, p):
                  r = self.rewards[x][y]
                  total += prob * (r + γ * V_s[x][y])

              ## Calculate the difference between old and new V values
              V_diff = max(V_diff, abs(total - V_s[i][j]))
              V_s[i][j] = total

          if V_diff < θ:
            break

        # Policy Improvement
        policy_stable = True
        ## Iterate through all states
        for i in range(self.length):
          for j in range(self.width):
            if self.board[i][j] in ['W', 'G']:
              continue
            old_action = optimal_policy[i][j]

            ## Update optimal policy
            Q_values = {}
            for action in self.actions.keys():
              total = 0
              for prob, (x, y) in self.get_next_states(i, j, action, p):
                  r = self.rewards[x][y]
                  total += prob * (r + γ * V_s[x][y])
              Q_values[action] = total

            best_a = max(Q_values, key=Q_values.get)
            optimal_policy[i][j] = best_a

            ## check if old action is equal to new policy
            if old_action != optimal_policy[i][j]:
              policy_stable = False
        # Exit loop if policy is stable or if loop exceeds T_max
        if policy_stable or (count>=T_max):
          break
      print(f'{count} iterations until convergence.')
      return V_s, optimal_policy

    def value_iteration_optimal_policy(self, p, γ, θ ):
      # Initialization
      V_s = np.zeros((self.length, self.width))
      count = 0

      # Value Iteration Loop
      while True:
        count += 1
        V_diff = 0
        ## Iterate through all states
        for i in range(self.length):
          for j in range(self.width):
            ## Skip walls and goal states
            if self.board[i][j] in ['W', 'G']:
              continue
            v = V_s[i][j]
            Q_values = {}
            for action in self.actions.keys():
              total = 0
              for prob, (x, y) in self.get_next_states(i, j, action, p):
                  r = self.rewards[x][y]
                  total += prob * (r + γ * V_s[x][y])
              Q_values[action] = total

            V_s[i][j] = max(Q_values.values())

            V_diff = max(V_diff, abs(v - V_s[i][j]))

        if V_diff < θ:
          break

      # Policy Improvement
      # Extract optimal policy from value function
      optimal_policy = np.full((self.length, self.width), '←')
      for i in range(self.length):
        for j in range(self.width):
          if self.board[i][j] not in ['W', 'G']:
            Q_values = {}
            for action in self.actions.keys():
              total = 0
              for prob, (x, y) in self.get_next_states(i, j, action, p):
                  r = self.rewards[x][y]
                  total += prob * (r + γ * V_s[x][y])
              Q_values[action] = total

            optimal_policy[i][j] = max(Q_values, key=Q_values.get)

      print(f'{count} iterations until convergence.')
      return V_s, optimal_policy
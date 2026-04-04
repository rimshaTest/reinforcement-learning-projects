import numpy as np

class biological_gene_network:
    GENE_TYPES = ['ATM', 'p53', 'WIP1', 'MDM2']
    ACTIONS = {
        'Nothing': np.array([0, 0, 0, 0]),
        'ATM':     np.array([1, 0, 0, 0]),
        'p53':     np.array([0, 1, 0, 0]),
        'WIP1':    np.array([0, 0, 1, 0]),
        'MDM2':    np.array([0, 0, 0, 1])
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
    
    def calculate_Avg_A(self, policy, n_episodes, episode_length):
        total_activation = 0

        for _ in range(n_episodes):
            # Random initial state
            s = np.random.randint(0, 16)
            episode_activation = 0

            for _ in range(episode_length):
                action_name = policy[s]
                M_a = self.transition_matrices[action_name]
                # Sample next state
                s = np.random.choice(16, p=M_a[s])
                episode_activation += np.sum(self.states[s])

            total_activation += episode_activation / episode_length

        return total_activation / n_episodes

    def value_iteration(self, γ, θ):
        # Initialization
        V = np.zeros(16)
        count = 0

        # Value Iteration
        while True:
            count += 1
            Q_values = np.zeros((16, len(self.ACTIONS)))
            for idx, action_name in enumerate(self.ACTIONS):
                R_a = self.reward_matrices[action_name]
                M_a = self.transition_matrices[action_name]
                Q_values[:, idx] = R_a + γ * (M_a @ V)

            V_new = np.max(Q_values, axis=1)
            if np.max(np.abs(V_new - V)) < θ:
                V = V_new
                break
            V = V_new

        print(f'{count} Iterations until convergence')
        # Extract policy
        policy_indices = np.argmax(Q_values, axis=1)
        policy = [list(self.ACTIONS.keys())[i] for i in policy_indices]
        
        return policy
    
    def policy_iteration(self, γ, θ):
        # Initialize policy with 'Nothing' action for all states
        action_keys = list(self.ACTIONS.keys())
        policy = ['Nothing'] * 16
        count = 0

        # Loop until the previous policy is equal to the current one
        while True:
            count += 1

            # Policy Evaluation
            # V^π = (I - γ * M(π))^(-1) * R(π)
            M_pi = np.array([self.transition_matrices[policy[i]][i] for i in range(16)])
            R_pi = np.array([self.reward_matrices[policy[i]][i] for i in range(16)])
            V = np.linalg.solve(np.eye(16) - γ * M_pi, R_pi)

            # Policy Improvement
            Q_values = np.zeros((16, len(self.ACTIONS)))
            for idx, action_name in enumerate(action_keys):
                R_a = self.reward_matrices[action_name]
                M_a = self.transition_matrices[action_name]
                Q_values[:, idx] = R_a + γ * (M_a @ V)

            new_policy_indices = np.argmax(Q_values, axis=1)
            new_policy = [action_keys[i] for i in new_policy_indices]

            if new_policy == policy:
                break
            policy = new_policy

        print(f'{count} Iterations until convergence')
        return policy
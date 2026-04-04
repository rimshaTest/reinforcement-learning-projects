import numpy as np

from project3_q2 import biological_gene_network
from project3_q1 import Maze


"""Problem 1"""
def run_problem_1():
    print("Running Problem 1...")
    penalties = {
        'wall': -0.8,
        'bump': -10,
        'oil': -5,
        'empty': -1,
        'goal': 200
    }

    walls = [
        (1, 4),
        (2, 4),
        (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15),
        (4, 2),
        (5, 2), (5, 5), (5, 8), (5, 14),
        (6, 2), (6, 5), (6, 8), (6, 11), (6, 12), (6, 13), (6, 14),
        (7, 5), (7, 8), (7, 14),
        (8, 5), (8, 8), (8, 14),
        (9, 0), (9, 1), (9, 2), (9, 3), (9, 5), (9, 8), (9, 9), (9, 14),
        (10, 5), (10, 9), (10, 12), (10, 14), (10, 15), (10, 16),
        (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 9), (11, 12), (11, 16),
        (12, 6), (12, 9), (12, 12), (12, 16),
        (13, 6), (13, 9), (13, 12),
        (14, 6), (14, 12), (14, 13), (14, 14), (14, 15),
        (16, 0), (16, 1), (16, 1), (16, 6), (16, 7), (16, 8), (16, 9), (16, 10), (16, 11)
        ]
    oils = [
        (1, 7), (1, 15),
        (3, 1),
        (4, 5),
        (9, 17),
        (14, 9),
        (15, 9),
        (16, 13), (16, 16),
        (17, 6)
    ]
    bumps = [
        (0, 10), (0, 11),
        (1, 0), (1, 1), (1, 2),
        (4, 0), (4, 8), (4, 16),
        (5, 16),
        (6, 1), (6, 9), (6, 10), (6, 16),
        (7, 16),
        (11, 10), (11, 11),
        (13, 0), (13, 1),
        (14, 16), (14, 17),
        (15, 6)
    ]
    goal = (2, 12)
    start = (14, 3)
    length, width = 18, 18

    p = 0.025
    γ = 0.96
    α = 0.25
    ϵ = 0.1
    num_of_episodes = 1000
    T_max = 1000
    β = 0.5
    q_all_rewards = []
    SARSA_all_rewards = []
    ac_all_rewards = []

    q_policies = []
    SARSA_policies = []
    ac_policies = []

    for i in range(10):
        print(f"Running iteration {i+1}/10...")
        maze = Maze(length, width, walls, oils, bumps, goal, start, penalties)
        Q_s_a, q_policy, q_rewards = maze.q_learning_optimal_policy(γ, α, ϵ, num_of_episodes, T_max)
        maze.print_board(boards=[q_policy], scenario="Q-Learning", type=f'policy_{i}', question="problem1")
        trajectory, cumulative_rewards = maze.sample_trajectory(q_policy, p, T_max)
        maze.print_board(trajectories=[trajectory], scenario="Q-Learning", type=f'trajectory_{i}', question="problem1")
        q_all_rewards.append(q_rewards)
        q_policies.append((maze, q_policy))

        maze = Maze(length, width, walls, oils, bumps, goal, start, penalties)
        SARSA_Q_s_a, SARSA_policy, SARSA_rewards = maze.SARSA_optimal_policy(γ, α, ϵ, num_of_episodes, T_max)
        maze.print_board(boards=[SARSA_policy], scenario="SARSA", type=f'policy_{i}', question="problem1")
        trajectory, cumulative_rewards = maze.sample_trajectory(SARSA_policy, p, T_max)
        maze.print_board(trajectories=[trajectory], scenario="SARSA", type=f'trajectory_{i}', question="problem1")
        SARSA_all_rewards.append(SARSA_rewards)
        SARSA_policies.append((maze, SARSA_policy))

        maze = Maze(length, width, walls, oils, bumps, goal, start, penalties)
        ac_V_s, ac_policy, ac_rewards = maze.actor_critic_optimal_policy(γ, α, β, num_of_episodes, T_max)
        maze.print_board(boards=[ac_policy], scenario="Actor-Critic", type=f'policy_{i}', question="problem1")
        trajectory, cumulative_rewards = maze.sample_trajectory(ac_policy, p, T_max)
        maze.print_board(trajectories=[trajectory], scenario="Actor-Critic", type=f'trajectory_{i}', question="problem1")
        ac_all_rewards.append(ac_rewards)
        ac_policies.append((maze, ac_policy))

    # Plot per algorithm
    maze.plot_avg_cumulative_rewards(all_rewards=[q_all_rewards], labels=['Q-Learning'], scenario="Q-Learning", type='avg_rewards', question="problem1")
    maze.plot_avg_cumulative_rewards(all_rewards=[SARSA_all_rewards], labels=['SARSA'], scenario="SARSA", type='avg_rewards', question="problem1")
    maze.plot_avg_cumulative_rewards(all_rewards=[ac_all_rewards], labels=['Actor-Critic'], scenario="Actor-Critic", type='avg_rewards', question="problem1")

    maze.plot_avg_cumulative_rewards(
        all_rewards=[q_all_rewards, SARSA_all_rewards, ac_all_rewards],
        labels=['Q-Learning', 'SARSA', 'Actor-Critic'],
        scenario="all_algorithms",
        type='avg_rewards_combined',
        question="problem1"
    )
    
    print("Running learning rate comparison for Q-Learning...")
    α_values = [0.05, 0.1, 0.25, 0.5]
    q_all_rewards = []
    for learning_rate in α_values:
        print(f"Testing α={learning_rate}...")
        q_α_all_rewards = []
        for i in range(10):
            print(f"  Iteration {i+1}/10...")
            maze = Maze(length, width, walls, oils, bumps, goal, start, penalties)
            Q_s_a, q_policy, q_rewards = maze.q_learning_optimal_policy(γ, learning_rate, ϵ, num_of_episodes, T_max)
            q_α_all_rewards.append(q_rewards)
        q_all_rewards.append(np.mean(np.array(q_α_all_rewards), axis=0))
    
    maze.plot_avg_cumulative_rewards(
        all_rewards=[[r] for r in q_all_rewards],
        labels=[f'α={learning_rate}' for learning_rate in α_values],
        scenario="all_learning_rates",
        type='avg_rewards_combined',
        question="problem1"
    )
    print("Problem 1 completed.")

"""Problem 2"""
def run_problem_2():
    print("Running Problem 2...")

    # Construct the controlled matrices
    p = 0.1
    γ = 0.9
    α = 0.25
    ϵ = 0.15
    n_episodes = 1000
    T_max = 100

    # Initialize the biological gene network environment
    λ = 0.95
    β = 0.05
    q_all_rewards = []
    SARSA_all_rewards = []
    ac_all_rewards = []

    q_policies = []
    SARSA_policies = []
    ac_policies = []
    with open(f"problem2_q_learning_policy.txt", "w") as f:
        f.write('')

    with open(f"problem2_SARSA_policy.txt", "w") as f:
        f.write('')

    with open(f"problem2_actor_critic_policy.txt", "w") as f:
        f.write('')

    for i in range(10):
        print(f"Running iteration {i+1}/10...")
        gene_network = biological_gene_network(p)
        Q_s_a, q_policy, q_rewards = gene_network.q_learning_optimal_policy(γ, α, ϵ, n_episodes, T_max)
        with open(f"problem2_q_learning_policy.txt", "a") as f:
            f.write(str(q_policy)+'\n')
        q_all_rewards.append(q_rewards)
        q_policies.append((gene_network, q_policy))

        gene_network = biological_gene_network(p)
        SARSA_Q_s_a, SARSA_policy, SARSA_rewards = gene_network.SARSA_optimal_policy(γ, α, ϵ, n_episodes, T_max)
        with open(f"problem2_SARSA_policy.txt", "a") as f:
            f.write(str(SARSA_policy)+'\n')
        SARSA_all_rewards.append(SARSA_rewards)
        SARSA_policies.append((gene_network, SARSA_policy))

        gene_network = biological_gene_network(p)
        ac_V_s, ac_policy, ac_rewards = gene_network.actor_critic_optimal_policy(γ, α, β, n_episodes, T_max)
        with open(f"problem2_actor_critic_policy.txt", "a") as f:
            f.write(str(ac_policy)+'\n')
        ac_all_rewards.append(ac_rewards)
        ac_policies.append((gene_network, ac_policy))

    # Plot per algorithm
    gene_network = biological_gene_network(p)
    gene_network.plot_avg_cumulative_rewards(all_rewards=[q_all_rewards], labels=['Q-Learning'], scenario="Q-Learning", type='avg_rewards', question="problem2")
    gene_network.plot_avg_cumulative_rewards(all_rewards=[SARSA_all_rewards], labels=['SARSA'], scenario="SARSA", type='avg_rewards', question="problem2")
    gene_network.plot_avg_cumulative_rewards(all_rewards=[ac_all_rewards], labels=['Actor-Critic'], scenario="Actor-Critic", type='avg_rewards', question="problem2")

    # Plot all algorithms together
    gene_network.plot_avg_cumulative_rewards(
        all_rewards=[q_all_rewards, SARSA_all_rewards, ac_all_rewards],
        labels=['Q-Learning', 'SARSA', 'Actor-Critic'],
        scenario="all_algorithms",
        type='avg_rewards_combined',
        question="problem2"
    )
    
    _, q_policy_for_vis = q_policies[0]
    _, sarsa_policy_for_vis = SARSA_policies[0]
    _, ac_policy_for_vis = ac_policies[0]

    q_counts = gene_network.execute_greedy_policy(q_policy_for_vis, n_episodes=100, T_max=T_max)
    sarsa_counts = gene_network.execute_greedy_policy(sarsa_policy_for_vis, n_episodes=100, T_max=T_max)
    ac_counts = gene_network.execute_greedy_policy(ac_policy_for_vis, n_episodes=100, T_max=T_max)

    gene_network.plot_visitation_counts(
        all_counts=[q_counts, sarsa_counts, ac_counts],
        labels=['Q-Learning', 'SARSA', 'Actor-Critic'],
        question="problem2"
    )

    print("Running SARSA-λ comparisons...")
    λ_values = [0, 0.5, 0.95]
    sarsa_lambda_all_rewards = []

    for λ_val in λ_values:
        print(f"Testing SARSA-λ with λ={λ_val}...")
        λ_rewards = []
        for i in range(10):
            gene_network = biological_gene_network(p)
            _, _, rewards = gene_network.SARSA_lambda_optimal_policy(γ, α, ϵ, λ_val, n_episodes, T_max)
            λ_rewards.append(rewards)
        sarsa_lambda_all_rewards.append(λ_rewards)

    gene_network.plot_avg_cumulative_rewards(
        all_rewards=sarsa_lambda_all_rewards,
        labels=[f'SARSA-λ (λ={λ_val})' for λ_val in λ_values],
        scenario="SARSA_lambda_comparison",
        type='SARSA_lambda_rewards',
        question="problem2"
    )

    print("Problem 2 completed.")
    print("=============================================================================")


if __name__ == "__main__":
    
    # Run Problem 1
    # run_problem_1()

    # Run Problem 2
    run_problem_2()
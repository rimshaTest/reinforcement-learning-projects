from project2_q1 import Maze
from project2_q2 import biological_gene_network
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

def perform_iteration(maze, method, p, γ, θ, scenario='output', n_trajectories=1, question=''):
    print(f"Performing {method} ({scenario}) with p={p}, γ={γ}, θ={θ}...")
    
    if method == 'PI':
        V_s, policy = maze.policy_iteration_optimal_policy(p, γ, θ)
    else:
        V_s, policy = maze.value_iteration_optimal_policy(p, γ, θ)

    if question in ['q1_1', 'q1_2']:
        maze.print_board(boards=[V_s], scenario=scenario, type='Values', question=question)
        maze.print_board(boards=[policy], scenario=scenario, type='Policies', question=question)
        path = maze.get_optimal_policy(policy)
        maze.print_board(boards=[path], scenario=scenario, type='optimal_policy', question=question)

    elif question == 'q1_3a':
        trajectories = [maze.sample_trajectory(policy, p)[0] for _ in range(n_trajectories)]
        maze.print_board(trajectories=trajectories, scenario=scenario, type='path_overlay', question=question)

    elif question == 'q1_3b':
        all_cumulative_rewards = [maze.sample_trajectory(policy, p)[1] for _ in range(n_trajectories)]
        return all_cumulative_rewards
    
    elif question == 'q1_4':
        maze.print_board(boards=[policy], scenario=scenario, type='policies', question=question)
        path = maze.get_optimal_policy(policy)
        maze.print_board(boards=[path], scenario=scenario, type='optimal_policy', question=question)

"""Problem 1"""
def run_problem_1():
    print("Running Problem 1...")
    maze = Maze(length, width, walls, oils, bumps, goal, start, penalties)

    """1. Policy Iteration (vector form)"""

    # Base scenario
    perform_iteration(maze, method='PI', p = 0.02, γ = 0.99, θ = 0.01, scenario='PI_base_scenario', question='q1_1')

    # Large Stochasticity Scenario
    perform_iteration(maze, method='PI', p = 0.4, γ = 0.99, θ = 0.01, scenario='PI_large_stochasticity', question='q1_1')

    # Small Discount Factor Scenario
    perform_iteration(maze, method='PI', p = 0.02, γ = 0.4, θ = 0.01, scenario='PI_small_discount_factor', question='q1_1')

    """2. Value Iteration (vector form)"""
    # Base scenario
    perform_iteration(maze, method='VI', p = 0.02, γ = 0.99, θ = 0.01, scenario='VI_base_scenario', question='q1_2')

    # Large Stochasticity Scenario
    perform_iteration(maze, method='VI', p = 0.4, γ = 0.99, θ = 0.01, scenario='VI_large_stochasticity', question='q1_2')

    # Small Discount Factor Scenario
    perform_iteration(maze, method='VI', p = 0.02, γ = 0.4, θ = 0.01, scenario='VI_small_discount_factor', question='q1_2')

    """3. Effect of stochasticity"""
    # a. Path Overlay
    p = [0.02, 0.2, 0.6]

    for prob in p:
        scenario = f'PI_path_overlay_p_{prob}'
        perform_iteration(maze, method='PI', p=prob, γ = 0.99, θ = 0.01, scenario=scenario, n_trajectories=2, question=f'q1_3a')

    # b. Average cumulative reward curves
    print("Calculating average cumulative rewards for different stochasticity levels...")
    p_values = [0.02, 0.2, 0.6]
    policies = []
    all_rewards = []

    for prob in p_values:
        V_s, policy = maze.policy_iteration_optimal_policy(prob, γ=0.99, θ=0.01)
        policies.append(policy)
        rewards = [maze.sample_trajectory(policy, prob)[1] for _ in range(10)]
        all_rewards.append(rewards)

    maze.plot_avg_cumulative_rewards(p_values, all_rewards=all_rewards, scenario='avg_cumulative_rewards', question='q1_3b')

    """4. Effect of bump penalty"""
    p = 0.02
    γ = 0.99
    θ = 0.01
    penalties_modified = penalties.copy()
    penalties_modified['bump'] = -50
    maze_high_bump = Maze(length, width, walls, oils, bumps, goal, start, penalties_modified)
    # Optimal policy plot and optimal path plot
    perform_iteration(maze_high_bump, method='PI', p=p, γ=γ, θ=θ, scenario='PI_high_bump_penalty', question='q1_4')

    print("Problem 1 completed.")
    print("=============================================================================")

"""Problem 2"""
def run_problem_2():
    print("Running Problem 2...")

    # Construct the controlled matrices
    γ = 0.99

    # Part a
    print("Part a")
    p = 0.045
    θ = 0.01
    # Initialize the biological gene network environment
    gene_network = biological_gene_network(p)
    policy = gene_network.value_iteration(γ, θ)
    print("Policy:", policy)
    n_episodes=75
    episode_length=150
    avg_a_base = gene_network.calculate_Avg_A(policy, n_episodes=n_episodes, episode_length=episode_length)
    avg_a_no_control = gene_network.calculate_Avg_A(['Nothing']*16, n_episodes=n_episodes, episode_length=episode_length)
    print("Average activation with control:", avg_a_base)
    print("Average activation without control:", avg_a_no_control)


    # Part b
    print("Part b")
    probabilities = [0.18, 0.55]
    for p in probabilities:
        print(f"Evaluating for p={p}...")
        gene_network = biological_gene_network(p)
        policy = gene_network.value_iteration(γ, θ)
        print("Policy:", policy)
        n_episodes=75
        episode_length=150
        avg_a_base = gene_network.calculate_Avg_A(policy, n_episodes=n_episodes, episode_length=episode_length)
        avg_a_no_control = gene_network.calculate_Avg_A(['Nothing']*16, n_episodes=n_episodes, episode_length=episode_length)
        print("Average activation with control:", avg_a_base)
        print("Average activation without control:", avg_a_no_control)


    # Part c
    print("Part c")
    p = 0.045
    gene_network = biological_gene_network(p)
    policy = gene_network.policy_iteration(γ, θ)
    print("Policy:", policy)

    print("Problem 2 completed.")
    print("=============================================================================")

if __name__ == "__main__":
    
    # Run Problem 1
    run_problem_1()

    # Run Problem 2
    run_problem_2()
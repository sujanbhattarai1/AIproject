import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from network import run_hopfield
from analysis import is_valid_tour, decode_tour, compute_tour_distance, two_opt
import config

def generate_random_cities(N, seed=42):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.1, 0.9, (N, 2))
    return coords

def run_experiment(run_idx, N, seed, time_const, step_size, penalty_row, penalty_col, penalty_distance, penalty_toursize):
    print(f"\n======================================")
    print(f"EXPERIMENT RUN {run_idx} (N={N}, time_const={time_const}, step_size={step_size})")
    print(f"Penalties: row/col={penalty_row}, dist={penalty_distance}, toursize={penalty_toursize}")
    print(f"==========================================")
    
    cities = generate_random_cities(N, seed)
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            distance_matrix[i][k] = np.linalg.norm(cities[i] - cities[k])
    
    if np.max(distance_matrix) > 0:
        distance_matrix = distance_matrix / np.max(distance_matrix)

    print(f"Running Hopfield...")
    
    activation, energy_history = run_hopfield(
        num_cities=N,
        distance_matrix=distance_matrix,
        random_seed=seed,
        penalty_row=penalty_row,
        penalty_col=penalty_col,
        penalty_distance=penalty_distance,
        penalty_toursize=penalty_toursize,
        step_size=step_size,
        time_const=time_const,
        max_iter=config.MAX_ITER
    )
    
    valid, binary_act = is_valid_tour(np.round(activation, 4), N)
    print(f"Tour Valid: {valid}")
    
    os.makedirs("Images", exist_ok=True)
    
    sns.set_theme(style="whitegrid")

    
    plt.figure(figsize=(8, 5))
    iterations = np.arange(len(energy_history)) * 100 
    
    sns.lineplot(x=iterations, y=energy_history, color='#e65100', linewidth=2)
    plt.yscale('log')
    plt.title(f"Hopfield Energy Convergence [Run {run_idx}]", fontsize=14, fontweight='bold')
    
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Lyapunov Energy (Log Scale)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"Images/energy_plot_run{run_idx}.pdf") 
    plt.savefig(f"Images/energy_plot_run{run_idx}.png")
    print(f"Saved -> Images/energy_plot_run{run_idx}.png")
    plt.close()

   
    plt.figure(figsize=(6, 6))
    
    sns.heatmap(activation, cmap='YlOrRd', annot=True, fmt=".2f", 
                cbar=False, square=True, annot_kws={"size": 8})
    
    plt.title(f"Final Activation Map [Run {run_idx}]", fontsize=14, fontweight='bold')
    plt.xlabel("Position in Tour", fontsize=12)
    plt.ylabel("City", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"Images/activation_map_run{run_idx}.pdf")
    plt.savefig(f"Images/activation_map_run{run_idx}.png")
    print(f"Saved -> Images/activation_map_run{run_idx}.png")
    plt.close()

    sns.reset_orig() 
    plt.clf()

    tour_indices = [int(i) for i in decode_tour(binary_act, N)]
    dist = compute_tour_distance(tour_indices, distance_matrix)
    
    status_title = "Pure Hopfield Tour" if valid else "FAILED Hopfield Tour"
    print(f"{status_title} Distance: {dist:.4f}")
    
    plt.figure(figsize=(6, 6))
    for i in range(len(tour_indices) - 1):
        c1 = cities[tour_indices[i]]
        c2 = cities[tour_indices[i + 1]]
        line_style = 'b-' if valid else 'r--'
        plt.plot([c1[0], c2[0]], [c1[1], c2[1]], line_style, linewidth=2, alpha=0.6)
        
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=150, zorder=5)
    for i, (x, y) in enumerate(cities):
        plt.text(x + 0.02, y + 0.02, chr(65 + i), fontsize=12, fontweight='bold')
        
    plt.title(f"{status_title} (Dist: {dist:.4f})]", 
              fontsize=14, fontweight='bold', color='black' if valid else 'red')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"Images/tour_pure_hopfield_run{run_idx}.pdf")
    plt.savefig(f"Images/tour_pure_hopfield_run{run_idx}.png")
    print(f"Saved -> Images/tour_pure_hopfield_run{run_idx}.png")
    plt.close()

    if valid:
        if N <= 9:
            print("Computing Exact Global Optimum...")
            min_dist = float('inf')
            for p in itertools.permutations(range(1, N)):
                tour = [0] + list(p)
                dist_val = sum(distance_matrix[tour[i], tour[(i+1)%N]] for i in range(N))
                if dist_val < min_dist:
                    min_dist = dist_val
            print(f"Exact True Optimal Distance: {min_dist:.4f}")
        else:
            print(f"N={N} is too large for fast exhaustive search. Skipping Exact Solver.")

        optimized_tour = two_opt(tour_indices, distance_matrix)
        opt_dist = compute_tour_distance(optimized_tour, distance_matrix)
        print(f"2-Opt Optimized Tour Distance: {opt_dist:.4f}")

        plt.figure(figsize=(6, 6))
        for i in range(len(optimized_tour) - 1):
            c1 = cities[optimized_tour[i]]
            c2 = cities[optimized_tour[i + 1]]
            plt.plot([c1[0], c2[0]], [c1[1], c2[1]], 'g-', linewidth=2, alpha=0.6)
            
        plt.scatter(cities[:, 0], cities[:, 1], c='red', s=150, zorder=5)
        for i, (x, y) in enumerate(cities):
            plt.text(x + 0.02, y + 0.02, chr(65 + i), fontsize=12, fontweight='bold')
            
        plt.title(f"2-Opt Optimized Tour (Dist: {opt_dist:.4f})]", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"Images/tour_2opt_run{run_idx}.pdf")
        plt.savefig(f"Images/tour_2opt_run{run_idx}.png")
        print(f"Saved -> Images/tour_2opt_run{run_idx}.png")
        plt.close()


def main():
    experiments = [
        {'N': 6, 'time_const': 1.0, 'step_size': 0.001, 'p_row': 1000, 'p_col': 1000, 'p_dist': 1.0, 'p_tour': 1000},
        {'N': 8, 'time_const': 1.0, 'step_size': 0.001, 'p_row': 500,  'p_col': 500,  'p_dist': 5.0, 'p_tour': 500},
        {'N': 9, 'time_const': 1.5, 'step_size': 0.002, 'p_row': 2000, 'p_col': 2000, 'p_dist': 0.5, 'p_tour': 2000}
    ]
    
    seed = np.random.randint(0, 100)
    
    for idx, exp in enumerate(experiments):
        run_experiment(
            run_idx=idx+1, N=exp['N'], seed=seed, 
            time_const=exp['time_const'], step_size=exp['step_size'],
            penalty_row=exp['p_row'], penalty_col=exp['p_col'], 
            penalty_distance=exp['p_dist'], penalty_toursize=exp['p_tour']
        )

if __name__ == "__main__":
    main()

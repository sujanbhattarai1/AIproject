import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from src.cities import generate_cities
from src.network import run_hopfield, compute_energy
from src.analysis import is_valid_tour

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
os.makedirs("plots", exist_ok=True)

#---------------------PENALTY SWEEP HEATMAP---------------------#
def experiment_penalty_sweep(
    num_cities          = 6,
    num_trials          = 15, 
    row_values          = None,
    dist_values         = None,
    penalty_col         = 500,
    penalty_toursize    = 500,
    step_size           = 0.001,
    time_const          = 1.0, 
    max_iter            = 20000,
):
    """
    Grid search over PENALY_ROW vs PENALTY_DISTANCE
    constraint strength vs distance optimiasation
    """
    if row_values is None: row_values = [200, 400, 600, 800, 1000]
    if dist_values is None: dist_values = [50, 100, 200, 400, 600]

    _, distance_matrix = generate_cities(num_cities, random_seed=0)

    print(f"\nExperiment 1 — penalty sweep  "
          f"({len(row_values)}x{len(dist_values)} grid, {num_trials} trials per cell)")

    grid = np.zeros((len(dist_values), len(row_values)))

    for ri, p_row in enumerate(row_values):
        for di, p_dist in enumerate(dist_values):
            successes = sum(
                1 for seed in range(num_trials)
                if is_valid_tour(
                    run_hopfield(num_cities, distance_matrix, seed, p_row, penalty_col, p_dist, penalty_toursize, step_size, time_const, max_iter)[0],
                    num_cities
                )[0]
            )
            grid[di, ri] = successes / num_trials
            print(f"  row={p_row:5d}  dist={p_dist:5d}  ->  {100*grid[di,ri]:.0f}%")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid, origin="upper", aspect="auto",
                    vmin=0, vmax=1, cmap="RdYlGn", interpolation="nearest")
        ax.set_xticks(range(len(row_values)));   ax.set_xticklabels(row_values)
        ax.set_yticks(range(len(dist_values)));  ax.set_yticklabels(dist_values)
        ax.set_xlabel("PENALTY_ROW  (constraint strength A)")
        ax.set_ylabel("PENALTY_DISTANCE  (distance optimisation C)")
        ax.set_title(
            f"Experiment 1 — Valid tour rate  (N={num_cities}, {num_trials} trials per cell)\n"
            "Green = valid tours found   |   Red = constraints violated",
            fontsize=11,
        )
        for ri in range(len(row_values)):
            for di in range(len(dist_values)):
                v  = grid[di, ri]
                tc = "white" if v < 0.25 or v > 0.75 else "black"
                ax.text(ri, di, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color=tc, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Valid tour rate")
    path = "plots/exp1_penalty_sweep.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return grid

#-----------------------SCALE VS SUCCESS RATE-----------------------#
def experiment_scale_vs_success(
    city_sizes       = None,
    num_trials       = 20,
    penalty_row      = 500,
    penalty_col      = 500,
    penalty_distance = 100,
    penalty_toursize = 500,
    step_size        = 0.001,
    time_const       = 1.0,
    max_iter         = 20000,
):
    if city_sizes is None:
        city_sizes = [4, 5, 6, 7, 8, 10, 12, 15]
 
    print(f"\nExperiment 2 — scale vs success  "
          f"(N = {city_sizes}, {num_trials} trials each)")
 
    rates = []
    for N in city_sizes:
        _, distance_matrix = generate_cities(N, random_seed=0)
        successes = sum(
            1 for seed in range(num_trials)
            if is_valid_tour(
                run_hopfield(N, distance_matrix, seed,
                             penalty_row, penalty_col, penalty_distance, penalty_toursize,
                             step_size, time_const, max_iter)[0],
                N
            )[0]
        )
        rate = successes / num_trials
        rates.append(rate)
        print(f"  N={N:3d}  ->  {successes}/{num_trials}  ({100*rate:.0f}%)")
 
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = sns.color_palette("muted")
    sns.lineplot(x=city_sizes, y=rates, marker="o",
                 color=palette[0], linewidth=2, markersize=7, ax=ax)
    for x, y in zip(city_sizes, rates):
        ax.annotate(f"{y:.0%}", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#555")
    ax.axhline(0.5, color=palette[3], linestyle="--", alpha=0.7,
               label="50% threshold")
    ax.set_xlabel("Number of cities  N")
    ax.set_ylabel("Valid tour rate")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(city_sizes)
    ax.set_title(
        f"Experiment 2 — Valid tour rate vs problem size  ({num_trials} trials per N)\n"
        "Classic Hopfield capacity limit: performance collapses around N > 8",
        fontsize=11,
    )
    ax.legend()
    path = "plots/exp2_scale_vs_success.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return city_sizes, rates

#--------------------LOCAL MINIMA DEMONSTRATION--------------------#
def experiment_local_minima(
    num_cities       = 6,
    num_trials       = 50,
    penalty_row      = 500,
    penalty_col      = 500,
    penalty_distance = 100,
    penalty_toursize = 500,
    step_size        = 0.001,
    time_const       = 1.0,
    max_iter         = 20000,
):
    print(f"\nExperiment 3 — local minima  (N={num_cities}, {num_trials} trials)")
 
    _, distance_matrix = generate_cities(num_cities, random_seed=0)
 
    valid_energies   = []
    invalid_energies = []
 
    for seed in range(num_trials):
        act, _ = run_hopfield(
            num_cities, distance_matrix, seed,
            penalty_row, penalty_col, penalty_distance, penalty_toursize,
            step_size, time_const, max_iter,
        )
        E = compute_energy(act, distance_matrix, num_cities,
                           penalty_row, penalty_col, penalty_distance, penalty_toursize)
        valid, _ = is_valid_tour(act, num_cities)
        (valid_energies if valid else invalid_energies).append(E)
 
    n_valid   = len(valid_energies)
    n_invalid = len(invalid_energies)
    print(f"  Valid:   {n_valid}/{num_trials}  ({100*n_valid//num_trials}%)")
    print(f"  Invalid: {n_invalid}/{num_trials}  ({100*n_invalid//num_trials}%)")
 
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = sns.color_palette("muted")
    all_e = valid_energies + invalid_energies
    bins  = np.linspace(min(all_e) * 0.85, max(all_e) * 1.05, 22)
    if valid_energies:
        ax.hist(valid_energies, bins=bins, alpha=0.75,
                color=palette[2], label=f"Valid tour  (n={n_valid})",
                edgecolor="white", linewidth=0.5)
    if invalid_energies:
        ax.hist(invalid_energies, bins=bins, alpha=0.65,
                color=palette[3], label=f"Stuck in local minimum  (n={n_invalid})",
                edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Final energy  E  at convergence")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Experiment 3 — Energy at convergence: valid tours vs local minima\n"
        f"(N={num_cities}, {num_trials} random seeds, same hyperparameters)",
        fontsize=11,
    )
    ax.legend()
    path = "plots/exp3_local_minima.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return valid_energies, invalid_energies

if __name__ == "__main__":
    print("=" * 60)
    print("Hopfield TSP — experiments")
    print("=" * 60)
    experiment_penalty_sweep()
    experiment_scale_vs_success()
    experiment_local_minima()
    print("\nAll done. Check plots/ for outputs:")
    print("  exp1_penalty_sweep.png")
    print("  exp2_scale_vs_success.png")
    print("  exp3_local_minima.png")

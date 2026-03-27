import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
 
 
def plot_tour(coordinates, tour, tour_distance, num_cities, valid, save_dir="plots"):
    """
    Plot 1 — the solved tour route.
    Cities as scatter points, arrows showing the path between them.
    Saved to plots/plot1_tour.png
    """
    os.makedirs(save_dir, exist_ok=True)
 
    fig, ax = plt.subplots(figsize=(6, 6))
 
    sns.scatterplot(
        x=coordinates[:, 0],
        y=coordinates[:, 1],
        s=150,
        color=sns.color_palette("muted")[0],   
        zorder=5,
        ax=ax
    )
 
    for idx, (x, y) in enumerate(coordinates):
        ax.annotate(
            chr(65 + idx), (x, y),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=13,
            fontweight="bold",
            color="#2d2d2d"
        )
 
    if valid:
        for step in range(num_cities):
            c1, c2 = tour[step], tour[step + 1]
            ax.annotate(
                "",
                xy=coordinates[c2],
                xytext=coordinates[c1],
                arrowprops=dict(
                    arrowstyle="->",
                    color=sns.color_palette("muted")[3],   
                    lw=2
                )
            )
        ax.set_title(f"Tour Route success distance = {tour_distance:.3f}", fontsize=12)
    else:
        ax.set_title("Tour Route invalid — tune penalties", fontsize=12)
 
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
 
    save_path = os.path.join(save_dir, "plot1_tour.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
 
 
def plot_energy(energy_history, save_dir="plots"):
    """
    Plot 2 — energy convergence over iterations.
    Shows E falling over time; a flat curve far above zero means local minimum.
    Saved to plots/plot2_energy.png
    """
    os.makedirs(save_dir, exist_ok=True)
 
    iterations = [i * 100 for i in range(len(energy_history))]
 
    fig, ax = plt.subplots(figsize=(7, 4))
 
    sns.lineplot(
        x=iterations,
        y=energy_history,
        color=sns.color_palette("muted")[0],
        lw=2,
        ax=ax
    )
 
    min_energy = min(energy_history)
    ax.axhline(
        min_energy,
        color=sns.color_palette("muted")[3],
        linestyle="--",
        alpha=0.8,
        label=f"min E = {min_energy:.1f}"
    )
 
    ax.set_title("Energy Convergence", fontsize=12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy  E")
    ax.legend()
 
    save_path = os.path.join(save_dir, "plot2_energy.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
 
 
def plot_activation_heatmap(activation, num_cities, save_dir="plots"):
    """
    Plot 3 — V matrix heatmap at convergence.
    A valid tour shows exactly one bright square per row and per column
    (like a permutation matrix).
    Saved to plots/plot3_heatmap.png
    """
    os.makedirs(save_dir, exist_ok=True)
 
    city_labels    = [chr(65 + i) for i in range(num_cities)]
    position_labels = [f"p{j + 1}" for j in range(num_cities)]
 
    fig, ax = plt.subplots(figsize=(5, 5))
 
    sns.heatmap(
        activation,
        ax=ax,
        cmap="YlGnBu",          
        vmin=0,
        vmax=1,
        annot=True,             
        fmt=".2f",
        linewidths=0.5,
        linecolor="#cccccc",
        xticklabels=position_labels,
        yticklabels=city_labels,
        cbar_kws={"label": "Activation"}
    )
 
    ax.set_title("Activation Matrix  V[city][position]", fontsize=12)
    ax.set_xlabel("Position in tour")
    ax.set_ylabel("City")
 
    save_path = os.path.join(save_dir, "plot3_heatmap.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
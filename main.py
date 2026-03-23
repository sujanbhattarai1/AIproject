import os
import numpy as np  
import seaborn as sns 
import config
from src.cities import generate_cities
from src.network import run_hopfield
from src.analysis import is_valid_tour, decode_tour, compute_tour_distance
from src.generate_plots import plot_tour, plot_energy, plot_activation_heatmap

def main():

    #-------------------------------SETUP-------------------------------#
    # make directory to hold plots if not exists
    os.makedirs("plots", exist_ok = True)
    coordinates, distance_matrix = generate_cities(config.NUM_CITIES, config.RANDOM_SEED)

    print(f"Running Hopfield TSP... {config.NUM_CITIES} cities")
    print(f"Penalties: row={config.PENALTY_ROW} col={config.PENALTY_COL} ")
    print(f"distance={config.PENALTY_DISTANCE}, toursize={config.PENALTY_TOURSIZE}")


    #----------------------------RUN NETWORK----------------------------#
    activation, energy_history = run_hopfield(
        num_cities = config.NUM_CITIES,
        distance_matrix = distance_matrix,
        random_seed = config.RANDOM_SEED,
        penalty_row = config.PENALTY_ROW,
        penalty_col = config.PENALTY_COL,
        penalty_distance = config.PENALTY_DISTANCE,
        penalty_toursize = config.PENALTY_TOURSIZE,
        step_size = config.STEP_SIZE,
        time_const = config.TIME_CONST,
        max_iter = config.MAX_ITER
    )

    #---------------------------DECODE RESULT---------------------------#
    valid, binary_activation = is_valid_tour(activation, config.NUM_CITIES
    )
    print(f"\nValid tour found: {valid}")

    tour = []
    tour_distance = 0.0
    if valid:
        tour = decode_tour(binary_activation, config.NUM_CITIES)
        tour_distance = compute_tour_distance(tour, distance_matrix)
        city_labels = [chr(65 + i) for i in tour]
        print(f"Tour order : {' -> '.join(city_labels)}")
        print(f"Tour distance: {tour_distance:.4f}")
    else:
        print("No valid tour...try increaseing PENALTY_ROW or PENALTY_COL,")
        print("or decreasing PENALTY_DISTANCE...")

    #----------------------------PLOT RESULTS----------------------------#
    plot_tour(coordinates, tour, tour_distance, config.NUM_CITIES, valid)
    plot_energy(energy_history)
    plot_activation_heatmap(activation, config.NUM_CITIES)


if __name__ == "__main__":
    main()
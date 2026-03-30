import numpy as np
from src.network import run_hopfield


def is_valid_tour(activation, num_cities, threshold=0.5):
    binary_activation = (activation > threshold).astype(int)

    each_city_visited_once = all(
        np.sum(binary_activation[i, :]) == 1 for i in range(num_cities)
    )
    one_city_per_slot = all(
        np.sum(binary_activation[:, j]) == 1 for j in range(num_cities)
    )

    return each_city_visited_once and one_city_per_slot, binary_activation


def decode_tour(binary_activation, num_cities):
    tour = []
    for position in range(num_cities):
        city = np.argmax(binary_activation[:, position])
        tour.append(city)
    tour.append(tour[0])
    return tour


def compute_tour_distance(tour, distance_matrix):
    return sum(
        distance_matrix[tour[step], tour[step + 1]]
        for step in range(len(tour) - 1)
    )


def two_opt(tour, distance_matrix):
    route    = list(tour[:-1])
    N        = len(route)
    improved = True

    while improved:
        improved = False
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                a, b = route[i - 1], route[i]
                c, d = route[j],     route[(j + 1) % N]

                if distance_matrix[a][c] + distance_matrix[b][d] < distance_matrix[a][b] + distance_matrix[c][d] - 1e-10:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    improved = True

    route.append(route[0])
    return route


def measure_success_rate(num_trials, num_cities, distance_matrix, penalty_row, penalty_col, penalty_distance, penalty_toursize, step_size, time_const, max_iter):
    successes = 0
    for trial in range(num_trials):
        activation, _ = run_hopfield(num_cities, distance_matrix, random_seed=trial, penalty_row=penalty_row, penalty_col=penalty_col, penalty_distance=penalty_distance, penalty_toursize=penalty_toursize, step_size=step_size, time_const=time_const, max_iter=max_iter)
        valid, _ = is_valid_tour(activation, num_cities)
        if valid:
            successes += 1

    print(f"Valid tours: {successes}/{num_trials}  ({100 * successes / num_trials:.0f}%)")
    return successes / num_trials
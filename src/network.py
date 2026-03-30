import numpy as np


def sigmoid(x, gain=1.0):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-gain * x))


def compute_energy(activation, distance_matrix, num_cities, penalty_row, penalty_col, penalty_distance, penalty_toursize):
    N = num_cities
    V = activation

    row_penalty      = penalty_row      / 2 * np.sum((np.sum(V, axis=1) - 1) ** 2)
    col_penalty      = penalty_col      / 2 * np.sum((np.sum(V, axis=0) - 1) ** 2)
    toursize_penalty = penalty_toursize / 2 * (np.sum(V) - N) ** 2

    V_next = np.roll(V, -1, axis=1)
    V_prev = np.roll(V,  1, axis=1)
    distance_penalty = penalty_distance / 2 * np.einsum('ij,ik,kj->', V, distance_matrix, V_next + V_prev)

    return row_penalty + col_penalty + distance_penalty + toursize_penalty


def compute_all_net_inputs(activation, distance_matrix, num_cities, penalty_row, penalty_col, penalty_distance, penalty_toursize):
    N = num_cities
    V = activation

    row_sums = np.sum(V, axis=1, keepdims=True)
    col_sums = np.sum(V, axis=0, keepdims=True)
    total    = np.sum(V)

    from_row      = -penalty_row      * (row_sums - V - 1)
    from_col      = -penalty_col      * (col_sums - V - 1)
    from_toursize = -penalty_toursize * (total - V - N)

    V_next = np.roll(V, -1, axis=1)
    V_prev = np.roll(V,  1, axis=1)
    from_distance = -penalty_distance * (distance_matrix @ (V_next + V_prev))

    return from_row + from_col + from_distance + from_toursize


def _clamp_to_permutation(V):
    N   = V.shape[0]
    out = np.zeros_like(V)

    row_winners = np.argmax(V, axis=1)
    out[np.arange(N), row_winners] = 1.0

    for col in range(N):
        rows_wanting_col = np.where(out[:, col] == 1)[0]
        if len(rows_wanting_col) > 1:
            best = rows_wanting_col[np.argmax(V[rows_wanting_col, col])]
            for r in rows_wanting_col:
                if r != best:
                    out[r, col] = 0
                    free_cols = np.where(out[:, :].sum(axis=0) == 0)[0]
                    if len(free_cols):
                        out[r, free_cols[0]] = 1
    return out


def run_hopfield(num_cities, distance_matrix, random_seed,
                 penalty_row, penalty_col, penalty_distance, penalty_toursize,
                 step_size, time_const, max_iter):
    N   = num_cities
    rng = np.random.default_rng(random_seed)

    def _init_state():
        perm = rng.permutation(N)
        mp   = np.zeros((N, N))
        for i, j in enumerate(perm):
            mp[i, j] = 1.0
        mp += rng.uniform(-0.05, 0.05, (N, N))
        return mp, sigmoid(mp)

    membrane_potential, activation = _init_state()

    theta         = 0.5
    epsilon_start = 0.15
    epsilon_end   = 0.0
    epsilon_decay = 0.99
    epsilon       = epsilon_start

    energy_history = []
    gain_start     = 0.5
    gain_end       = 20.0

    best_energy         = float("inf")
    steps_since_improve = 0
    plateau_limit       = 600
    restarts            = 0
    max_restarts        = 2

    for step in range(max_iter):
        progress = step / max_iter
        gain     = gain_start * (gain_end / gain_start) ** progress

        net = compute_all_net_inputs(
            activation, distance_matrix, N,
            penalty_row, penalty_col, penalty_distance, penalty_toursize
        )

        net = net + theta

        if epsilon > epsilon_end:
            net = net + rng.uniform(-epsilon, epsilon, (N, N))

        membrane_potential += step_size * (-membrane_potential / time_const + net)
        activation          = sigmoid(membrane_potential, gain=gain)

        if step > max_iter * 0.3 and step % 500 == 0:
            hard               = _clamp_to_permutation(activation)
            activation         = 0.9 * activation + 0.1 * hard
            membrane_potential = np.log(np.clip(activation, 1e-6, 1 - 1e-6)) / gain

        if step % 100 == 0:
            energy = compute_energy(
                activation, distance_matrix, N,
                penalty_row, penalty_col, penalty_distance, penalty_toursize
            )
            energy_history.append(float(energy))

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if energy < best_energy - 1e-3:
                best_energy         = energy
                steps_since_improve = 0
            else:
                steps_since_improve += 100

            if steps_since_improve >= plateau_limit and restarts < max_restarts:
                membrane_potential, activation = _init_state()
                epsilon             = epsilon_start * (0.5 ** restarts)
                steps_since_improve = 0
                restarts           += 1

    return activation, energy_history

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import config
import sys, os
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from network import run_hopfield, compute_energy
from analysis import is_valid_tour, decode_tour, compute_tour_distance, two_opt

NUM_TRIALS_MAX = 15
NUM_KEEP_BEST  = 3
MAX_WORKERS    = 4

app = Flask(__name__)
CORS(app)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


def _run_trial(args):
    (N, distance_matrix, seed,
     penalty_row, penalty_col, penalty_distance, penalty_toursize,
     step_size, time_const, max_iter) = args

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from network import run_hopfield
    from analysis import is_valid_tour, decode_tour, compute_tour_distance, two_opt
    import numpy as np

    act, e_hist = run_hopfield(
        num_cities       = N,
        distance_matrix  = distance_matrix,
        random_seed      = seed,
        penalty_row      = penalty_row,
        penalty_col      = penalty_col,
        penalty_distance = penalty_distance,
        penalty_toursize = penalty_toursize,
        step_size        = step_size,
        time_const       = time_const,
        max_iter         = max_iter,
    )

    valid, binary_act = is_valid_tour(np.round(act, 4), N)

    if valid:
        t   = [int(i) for i in decode_tour(binary_act, N)]
        d   = float(compute_tour_distance(t, distance_matrix))
        t2  = [int(i) for i in two_opt(t, distance_matrix)]
        d2  = float(compute_tour_distance(t2, distance_matrix))
        return {
            "valid":         True,
            "activation":    act,
            "energyHistory": e_hist,
            "binary":        binary_act,
            "tour":          t2,
            "tourDistance":  d2,
            "rawDistance":   d,
            "seed":          seed,
        }
    else:
        bin_act    = (act > 0.5).astype(int)
        violations = int(np.sum(np.abs(bin_act.sum(axis=1) - 1)) +
                        np.sum(np.abs(bin_act.sum(axis=0) - 1)))
        return {
            "valid":         False,
            "activation":    act,
            "energyHistory": e_hist,
            "binary":        bin_act,
            "violations":    violations,
            "seed":          seed,
        }


@app.route("/solve", methods=["POST"])
def solve():
    body = request.get_json(force=True)

    raw_cities = body.get("cities", [])
    if len(raw_cities) < 3:
        return jsonify({"error": "need at least 3 cities"}), 400

    N           = len(raw_cities)
    coordinates = np.array([[c["x"], c["y"]] for c in raw_cities])

    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            distance_matrix[i][k] = np.linalg.norm(coordinates[i] - coordinates[k])
    if np.max(distance_matrix) > 0:
        distance_matrix = distance_matrix / np.max(distance_matrix)

    penalty_row      = int(body.get("penaltyRow",      config.PENALTY_ROW))
    penalty_col      = int(body.get("penaltyCol",      config.PENALTY_COL))
    penalty_distance = float(body.get("penaltyDistance", config.PENALTY_DISTANCE))
    penalty_toursize = int(body.get("penaltyToursize", config.PENALTY_TOURSIZE))
    max_iter         = int(body.get("maxIter",         config.MAX_ITER))
    step_size        = float(body.get("stepSize",      config.STEP_SIZE))
    time_const       = float(body.get("timeConst",     config.TIME_CONST))
    random_seed      = int(body.get("randomSeed",      config.RANDOM_SEED))

    trial_args = [
        (N, distance_matrix, random_seed + i,
         penalty_row, penalty_col, penalty_distance, penalty_toursize,
         step_size, time_const, max_iter)
        for i in range(NUM_TRIALS_MAX)
    ]

    valid_results   = []
    invalid_results = []
    trials_run      = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_trial, a): a for a in trial_args}
        for future in as_completed(futures):
            trials_run += 1
            result = future.result()
            if result["valid"]:
                valid_results.append(result)
                if len(valid_results) >= NUM_KEEP_BEST:
                    for f in futures:
                        f.cancel()
                    break
            else:
                invalid_results.append(result)

    if valid_results:
        best = min(valid_results, key=lambda r: r["tourDistance"])

        activation          = best["activation"]
        energy_history      = best["energyHistory"]
        binary_activation   = best["binary"]
        valid               = True
        tour                = best["tour"]
        tour_distance       = best["tourDistance"]
        raw_distance        = best.get("rawDistance", tour_distance)
        two_opt_improvement = round((1 - tour_distance / raw_distance) * 100, 2) if raw_distance > 0 else 0.0
        tour_labels         = [chr(65 + i) for i in tour]
        winning_seed        = best["seed"]
        diagnostics         = []

    else:
        best = min(invalid_results, key=lambda r: r["violations"])

        activation          = best["activation"]
        energy_history      = best["energyHistory"]
        binary_activation   = best["binary"]
        valid               = False
        tour                = []
        tour_distance       = 0.0
        raw_distance        = 0.0
        two_opt_improvement = 0.0
        tour_labels         = []
        winning_seed        = best["seed"]

        bin_act  = binary_activation
        row_sums = bin_act.sum(axis=1)
        col_sums = bin_act.sum(axis=0)
        diagnostics = []
        for i, s in enumerate(row_sums):
            if s != 1:
                diagnostics.append(f"city {chr(65+i)} visited {s}x — increase row penalty")
        for j, s in enumerate(col_sums):
            if s != 1:
                diagnostics.append(f"position {j+1} has {s} cities — increase col penalty")
        total = int(bin_act.sum())
        if total != N:
            diagnostics.append(f"{total} active neurons, expected {N} — adjust tour size penalty")
        diagnostics.append(f"tried {trials_run} trials — none produced a valid tour")
        if not diagnostics:
            diagnostics.append("no obvious constraint violations — try increasing distance penalty")

    return jsonify({
        "valid":             valid,
        "tour":              tour,
        "tourLabels":        tour_labels,
        "tourDistance":      tour_distance,
        "twoOptImprovement": two_opt_improvement,
        "activation":        activation.tolist(),
        "energyHistory":     energy_history,
        "diagnostics":       diagnostics,
        "numCities":         N,
        "trialsRun":         trials_run,
        "winningSeed":       winning_seed,
    })


if __name__ == "__main__":
    print("Hopfield TSP server running at http://localhost:5000")
    print("Open index.html in your browser")
    app.run(host="0.0.0.0", port=5000, debug=True)

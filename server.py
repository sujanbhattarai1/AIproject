from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import config
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from network import run_hopfield, compute_energy
from analysis import is_valid_tour, decode_tour, compute_tour_distance

app = Flask(__name__)
CORS(app)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/solve", methods=["POST"])
def solve():
    body = request.get_json(force=True)

    raw_cities = body.get("cities", [])
    if len(raw_cities) < 3:
        return jsonify({"error": "need at least 3 cities"}), 400

    N = len(raw_cities)
    coordinates = np.array([[c["x"], c["y"]] for c in raw_cities])

    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            distance_matrix[i][k] = np.linalg.norm(coordinates[i] - coordinates[k])
    if np.max(distance_matrix) > 0:
        distance_matrix = distance_matrix / np.max(distance_matrix) # Normalization of distance_matrix

    penalty_row      = int(body.get("penaltyRow",      config.PENALTY_ROW))
    penalty_col      = int(body.get("penaltyCol",      config.PENALTY_COL))
    penalty_distance = float(body.get("penaltyDistance", config.PENALTY_DISTANCE))
    penalty_toursize = int(body.get("penaltyToursize", config.PENALTY_TOURSIZE))
    max_iter         = int(body.get("maxIter",         config.MAX_ITER))
    step_size        = float(body.get("stepSize",      config.STEP_SIZE))
    time_const       = float(body.get("timeConst",     config.TIME_CONST))
    random_seed      = int(body.get("randomSeed",      config.RANDOM_SEED))

    activation, energy_history = run_hopfield(
        num_cities       = N,
        distance_matrix  = distance_matrix,
        random_seed      = random_seed,
        penalty_row      = penalty_row,
        penalty_col      = penalty_col,
        penalty_distance = penalty_distance,
        penalty_toursize = penalty_toursize,
        step_size        = step_size,
        time_const       = time_const,
        max_iter         = max_iter,
    )

    activation_rounded = np.round(activation, 4)
    valid, binary_activation = is_valid_tour(activation_rounded, N)

    tour          = []
    tour_distance = 0.0
    tour_labels   = []

    if valid:
        tour          = [int(i) for i in decode_tour(binary_activation, N)]
        tour_distance = float(compute_tour_distance(tour, distance_matrix))
        tour_labels   = [chr(65 + i) for i in tour]

    diagnostics = []
    if not valid:
        bin_act  = (activation > 0.5).astype(int)
        row_sums = bin_act.sum(axis=1)
        col_sums = bin_act.sum(axis=0)
        for i, s in enumerate(row_sums):
            if s != 1:
                diagnostics.append(f"city {chr(65+i)} visited {s}x — increase row penalty")
        for j, s in enumerate(col_sums):
            if s != 1:
                diagnostics.append(f"position {j+1} has {s} cities — increase col penalty")
        total = int(bin_act.sum())
        if total != N:
            diagnostics.append(f"{total} active neurons, expected {N} — adjust tour size penalty")
        if not diagnostics:
            diagnostics.append("no obvious constraint violations — try increasing distance penalty")

    return jsonify({
        "valid":         valid,
        "tour":          tour,
        "tourLabels":    tour_labels,
        "tourDistance":  tour_distance,
        "activation":    activation.tolist(),
        "energyHistory": energy_history,
        "diagnostics":   diagnostics,
        "numCities":     N,
    })


if __name__ == "__main__":
    print("Hopfield TSP server running at http://localhost:5000")
    print("Open index.html in your browser")
    app.run(host="0.0.0.0", port=5000, debug=True)

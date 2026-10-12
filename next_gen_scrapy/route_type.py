"""
Route type from the shape of a route-chart line, for match_to_pbp.

NGS participation data names the route the targeted receiver ran (`target_route`: GO, SLANT, OUT, ...). A chart
line's shape says much the same thing, so for every candidate play the matcher can ask how likely this line is
to be that play's route. That separates a receiver's targets far better than depth alone - two 8-yard catches
are rarely both slants.

The model is a small multilayer perceptron over compact shape features (route_type_model.npz, a few hundred kB,
evaluated with numpy alone). Trained on confidently matched routes, it names a held-out season's route right
~68% of the time; the matcher uses its probabilities, not its guess, so the near-misses (post vs in, corner vs
out) still help. Retrain with fit().
"""
import os

import numpy as np

N_PTS = 16
CLASSES = ["ANGLE", "CORNER", "CROSS", "FLAT", "GO", "HITCH", "IN", "OUT", "POST", "SCREEN", "SLANT", "WHEEL"]
# nflverse renamed / split routes in 2023: both eras map onto this shared set
UNIFY = {"HITCH/CURL": "HITCH", "IN/DIG": "IN", "SHALLOW CROSS/DRAG": "CROSS", "QUICK OUT": "OUT",
         "DEEP OUT": "OUT", "SWING": "FLAT", "TEXAS/ANGLE": "ANGLE"}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "route_type_model.npz")
_MODEL = None


def features(px, py, seg, complete, era2023):
    """Shape features of one route line (field yd), mirrored so +x is toward the receiver's own sideline."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    pre = np.asarray(seg, object) != "after_catch" if seg is not None else np.ones(len(px), bool)
    x, y, ax, ay = px[pre], py[pre], px[~pre], py[~pre]
    f = np.full(2 * N_PTS + 16, np.nan)
    if len(x) < 2:
        return f
    flip = 1.0 if x[0] >= 0 else -1.0
    xm, axm = x * flip, ax * flip
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xm), np.diff(y)))])
    L = s[-1]
    g = np.linspace(0, L, N_PTS) if L > 0 else np.zeros(N_PTS)
    f[:N_PTS], f[N_PTS:2 * N_PTS] = np.interp(g, s, xm) - xm[0], np.interp(g, s, y) - y[0]
    k_end = max(0, np.searchsorted(s, L - 3))                  # heading over the last ~3 yd before the catch
    k_rel = min(len(s) - 1, np.searchsorted(s, min(5, L)))      # ...and over the first ~5 yd (the release)
    h_end = np.arctan2(xm[-1] - xm[k_end], y[-1] - y[k_end])
    h_rel = np.arctan2(xm[k_rel] - xm[0], y[k_rel] - y[0])
    k_ac = min(len(ax) - 1, 6)                                  # which way the after-catch run went
    f[2 * N_PTS:] = [L, abs(x[0]), y[0], y[-1], xm[-1] - xm[0], y.max(), (xm - xm[0]).max(), (xm - xm[0]).min(),
                     np.sin(h_end), np.cos(h_end), np.sin(h_rel), np.cos(h_rel),
                     (axm[k_ac] - axm[0]) if len(ax) >= 2 else 0.0, (ay[k_ac] - ay[0]) if len(ax) >= 2 else 0.0,
                     float(complete), float(era2023)]
    return f


def feature_matrix(paths):
    """One feature row per route of a to_paths()-style frame (path_x, path_y, path_segment, route_type, season)."""
    rows = []
    for r in paths.itertuples(index=False):
        seg = getattr(r, "path_segment", None)
        seg = None if seg is None or isinstance(seg, float) else seg
        rows.append(features(r.path_x, r.path_y, seg, getattr(r, "route_type", "COMPLETE") == "COMPLETE",
                             int(getattr(r, "season", 2023)) >= 2023))
    return np.array(rows) if rows else np.zeros((0, 2 * N_PTS + 16))


def _load(path=None):
    global _MODEL
    if path is not None:
        z = np.load(path)
        return {k: z[k] for k in z.files}
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            return None
        z = np.load(MODEL_PATH)
        _MODEL = {k: z[k] for k in z.files}
    return _MODEL


def predict_proba(paths, model=None):
    """(n_routes, len(CLASSES)) route-type probabilities; rows of NaN where the line has < 2 points before the catch
    (or when no model is available)."""
    X = feature_matrix(paths)
    m = model if isinstance(model, dict) else _load(model)
    P = np.full((len(X), len(CLASSES)), np.nan)
    if m is None or not len(X):
        return P
    ok = np.isfinite(X).all(1)
    h = (X[ok] - m["mu"]) / m["sd"]
    n_layers = int(m["n_layers"])
    for i in range(n_layers):
        h = h @ m[f"W{i}"] + m[f"b{i}"]
        if i < n_layers - 1:
            h = np.maximum(h, 0.0)                              # relu
    h = h - h.max(1, keepdims=True)
    e = np.exp(h)
    P[ok] = e / e.sum(1, keepdims=True)
    return P


def fit(matched, path=MODEL_PATH, min_match_prob=0.5, hidden=(128, 64), seed=0):
    """Train on match_to_pbp(..., 'route') output (rows with a target_route and match_prob above the floor) and
    save the weights. Needs scikit-learn. Returns the model dict."""
    from sklearn.neural_network import MLPClassifier
    d = matched[matched["match_prob"] > min_match_prob]
    lab = d["target_route"].replace(UNIFY)
    d = d[lab.isin(CLASSES)]
    y = lab[lab.isin(CLASSES)].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    X = feature_matrix(d)
    ok = np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = MLPClassifier(hidden, max_iter=300, early_stopping=True, random_state=seed, alpha=1e-3).fit((X - mu) / sd, y)
    if list(clf.classes_) != list(range(len(CLASSES))):
        raise ValueError("training data must contain every route class")
    m = {"mu": mu, "sd": sd, "n_layers": np.array(len(clf.coefs_)),
         "prior": np.bincount(y, minlength=len(CLASSES)) / len(y)}
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        m[f"W{i}"], m[f"b{i}"] = W, b
    if path:
        np.savez_compressed(path, **m)
    return m

"""
What a carry chart line says about its run, as play-by-play would record it: which gap it went through (left/right
end, tackle, guard, or the middle) and whether it was run from shotgun - read from the line's whole shape (how it
starts, how it hits the line of scrimmage, where it goes), not one crossing point.

A small neural net per question, trained on carries matched to their play with high confidence; the weights ship
as carry_type_model.npz. match_to_pbp uses the probabilities as evidence (see pbp_match._carry_type_cost).

A third net reads how long the play lasted (snap to whistle, in seconds) off the line - mostly its length, how much
of it is behind the line of scrimmage and how often it turns. Play-by-play records that time for every play since
2022 (time_of_day -> end_clock_time), so it tells apart runs that gained the same yards in the same direction.
"""
import os

import numpy as np

N_PTS = 16
GAPS = ["left/end", "left/tackle", "left/guard", "middle/", "right/guard", "right/tackle", "right/end"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "carry_type_model.npz")
_MODEL = None


def features(px, py):
    px, py = np.asarray(px, float), np.asarray(py, float)
    d = np.r_[0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
    L = d[-1] if d[-1] > 0 else 1.0
    t = np.linspace(0, L, N_PTS)
    rx, ry = np.interp(t, d, px) - px[0], np.interp(t, d, py) - py[0]
    # up to 3 yds past the line of scrimmage, resampled along the path: how the run hits the line
    k = np.flatnonzero(py >= min(3.0, py.max()))
    k = k[0] if len(k) else len(py) - 1
    dd = d[:k + 1]
    tt = np.linspace(0, dd[-1] if dd[-1] > 0 else 1.0, 8)
    ex, ey = np.interp(tt, dd, px[:k + 1]) - px[0], np.interp(tt, dd, py[:k + 1])
    cr = np.flatnonzero((py[:-1] < 0) & (py[1:] >= 0))
    if len(cr):
        i = cr[0]
        cx, has = px[i] + (-py[i]) / (py[i + 1] - py[i]) * (px[i + 1] - px[i]) - px[0], 1.0
    else:
        cx, has = 0.0, 0.0
    return np.r_[rx, ry, ex, ey, px[0], py[0], px[-1] - px[0], py[-1], cx, has, L, px.min() - px[0],
                 px.max() - px[0], py.min()]


DUR_FEATURES = ["length", "length_behind_los", "length_past_los", "y0", "x0", "y_end", "dx", "abs_dx", "lateral",
                "turning", "n_turns", "n_points", "abs_x_end", "y_max", "y_min", "is_qb"]


def duration_features(px, py, is_qb=False):
    """What a line says about how long its play took (see DUR_FEATURES)."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    seg = np.hypot(np.diff(px), np.diff(py))
    L = seg.sum()
    behind = seg[py[:-1] < 0].sum()
    ang = np.arctan2(np.diff(py), np.diff(px))
    turn = np.abs(np.angle(np.exp(1j * np.diff(ang)))) if len(ang) > 1 else np.zeros(1)
    return np.r_[L, behind, L - behind, py[0], px[0], py[-1], px[-1] - px[0], abs(px[-1] - px[0]), np.abs(np.diff(px)).sum(),
                 turn.sum(), (turn > 0.5).sum(), len(px), abs(px[-1]), py.max(), py.min(), float(is_qb)]


def feature_matrix(paths):
    rows = [features(px, py) if px is not None and len(px) >= 2 else np.full(2 * N_PTS + 26, np.nan)
            for px, py in zip(paths["path_x"], paths["path_y"])]
    return np.array(rows) if rows else np.zeros((0, 2 * N_PTS + 26))


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


def _forward(m, head, X):
    h = (X - m[f"{head}_mu"]) / m[f"{head}_sd"]
    n = int(m[f"{head}_n_layers"])
    for i in range(n):
        h = h @ m[f"{head}_W{i}"] + m[f"{head}_b{i}"]
        if i < n - 1:
            h = np.maximum(h, 0.0)
    if head == "dur":                       # regression: seconds
        return h[:, 0]
    if h.shape[1] == 1:                     # a two-class net has one logistic output
        p1 = 1.0 / (1.0 + np.exp(-h[:, 0]))
        return np.column_stack([1 - p1, p1])
    h = h - h.max(1, keepdims=True)
    e = np.exp(h)
    return e / e.sum(1, keepdims=True)


def predict_proba(paths, model=None):
    """{'gap': (n, len(GAPS)), 'shotgun': (n, 2) [not, shotgun]} probabilities; NaN rows where the line is too short
    (or no model is available)."""
    X = feature_matrix(paths)
    m = model if isinstance(model, dict) else _load(model)
    out = {"gap": np.full((len(X), len(GAPS)), np.nan), "shotgun": np.full((len(X), 2), np.nan)}
    if m is None or not len(X):
        return out
    ok = np.isfinite(X).all(1)
    for head in out:
        out[head][ok] = _forward(m, head, X[ok])
    return out


def predict_duration(paths, model=None):
    """Expected snap-to-whistle seconds of each line's play (NaN where the line is too short or no model)."""
    m = model if isinstance(model, dict) else _load(model)
    n = len(paths)
    out = np.full(n, np.nan)
    if m is None or "dur_n_layers" not in m or not n:
        return out
    qb = paths["position"].eq("QB").to_numpy() if "position" in paths else np.zeros(n, bool)
    X = np.array([duration_features(px, py, q) if px is not None and len(px) >= 2 else np.full(len(DUR_FEATURES), np.nan)
                  for px, py, q in zip(paths["path_x"], paths["path_y"], qb)])
    ok = np.isfinite(X).all(1)
    if ok.any():
        out[ok] = _forward(m, "dur", X[ok])
    return out


def play_duration(pbp_like):
    """Seconds from snap to the end of the play, from play-by-play's time_of_day / end_clock_time (2022+); NaN where
    either is missing or the gap is implausible."""
    if "time_of_day" not in pbp_like or "end_clock_time" not in pbp_like:
        return np.full(len(pbp_like), np.nan)
    import pandas as pd
    t0 = pd.to_datetime(pbp_like["time_of_day"], utc=True, errors="coerce", format="mixed")
    t1 = pd.to_datetime(pbp_like["end_clock_time"], utc=True, errors="coerce", format="mixed")
    d = (t1 - t0).dt.total_seconds().to_numpy(float)
    return np.where((d >= 1.0) & (d <= 20.0), d, np.nan)


def labels(pbp_like):
    """pbp rows -> (gap class index or -1, shotgun 0/1 or -1)."""
    loc = pbp_like["run_location"].fillna("")
    gap = loc + "/" + np.where(loc == "middle", "", pbp_like["run_gap"].fillna("") if "run_gap" in pbp_like else "")
    g = gap.map({c: i for i, c in enumerate(GAPS)}).fillna(-1).astype(int).to_numpy()
    s = pbp_like["shotgun"].fillna(-1).astype(int).to_numpy() if "shotgun" in pbp_like else np.full(len(pbp_like), -1)
    return g, s


def fit(matched, path=MODEL_PATH, min_match_prob=0.8, hidden=(128, 64), seed=0):
    """Train on match_to_pbp(..., 'carry') output (confident matches) and save the weights. Needs scikit-learn."""
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    d = matched[matched["match_prob"] >= min_match_prob]
    X = feature_matrix(d)
    g, s = labels(d)
    m = {}
    for head, y, k in (("gap", g, len(GAPS)), ("shotgun", s, 2)):
        ok = np.isfinite(X).all(1) & (y >= 0)
        Xh, yh = X[ok], y[ok]
        mu, sd = Xh.mean(0), Xh.std(0) + 1e-6
        clf = MLPClassifier(hidden, max_iter=400, early_stopping=True, random_state=seed, alpha=1e-3).fit((Xh - mu) / sd, yh)
        if list(clf.classes_) != list(range(k)):
            raise ValueError(f"training data must contain every {head} class")
        m.update({f"{head}_mu": mu, f"{head}_sd": sd, f"{head}_n_layers": np.array(len(clf.coefs_)),
                  f"{head}_prior": np.bincount(yh, minlength=k) / len(yh)})
        for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
            m[f"{head}_W{i}"], m[f"{head}_b{i}"] = W, b
    # duration: plays with a recorded time, leaving out what the line can't show (QB scrambles' dropback, fumbles)
    dur = play_duration(d)
    keep = np.isfinite(dur)
    for col in ("qb_scramble", "fumble"):
        if col in d:
            keep &= d[col].fillna(0).to_numpy() != 1
    if keep.sum() >= 200:
        dd = d[keep]
        qb = dd["position"].eq("QB").to_numpy() if "position" in dd else np.zeros(len(dd), bool)
        X = np.array([duration_features(px, py, q) for px, py, q in zip(dd["path_x"], dd["path_y"], qb)])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        reg = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=600, early_stopping=True, random_state=seed,
                           learning_rate_init=2e-3).fit((X - mu) / sd, np.clip(dur[keep], 1.5, 12))
        m.update({"dur_mu": mu, "dur_sd": sd, "dur_n_layers": np.array(len(reg.coefs_))})
        for i, (W, b) in enumerate(zip(reg.coefs_, reg.intercepts_)):
            m[f"dur_W{i}"], m[f"dur_b{i}"] = W, b
    if path:
        np.savez_compressed(path, **m)
    return m

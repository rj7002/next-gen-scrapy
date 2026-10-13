"""
A learned scorer for (carry chart line, candidate run) pairs, for match_to_pbp.

The hand-built carry terms each compare one thing about a line with one pbp column. This model looks at a line and a
run together - every chart feature the matcher computes, every usable play-by-play / FTN / participation column, and
the key differences (line end vs yards, line length vs NGS distance, start vs the snap's hash, end vs the next snap's
hash) - and learns which combinations tell a back's real run from his look-alike runs that game. It is a
gradient-boosted tree ensemble trained on confidently matched carries (2019-22) and shipped as plain arrays
(carry_pair_model.npz), evaluated with numpy alone. On held-out 2024-25 runs it moves the matched line's end within
3 yds of the tracked end from 80.4% to 82.1% (Big Data Bowl tackle spots).

Retrain with fit().
"""
import json
import os

import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "carry_pair_model.npz")
CAT_COLS = ["run_location", "run_gap", "starting_hash", "qb_location", "offense_formation", "offense_personnel",
            "defense_personnel", "defense_coverage_type", "defense_man_zone_type", "posteam_type", "roof", "surface",
            "play_type_nfl", "next_hash_carry"]
DROP_SUBSTR = ("_id", "_name", "player", "game_id", "play_id", "old_game", "nflverse", "jersey", "_team", "stadium",
               "coach", "referee", "weather", "time", "date", "desc", "season", "week", "home_", "away_", "_score",
               "spread", "total_line")
_HASH = {"L": -3.08, "M": 0.0, "R": 3.08}
_SIDE = {"L": -1.0, "M": 0.0, "R": 1.0}
_MODEL = None


def pbp_numeric_columns(pbp):
    """The numeric pbp / FTN / participation columns the model may use (ids, names, clocks and scores left out)."""
    return [c for c in pbp.columns if pd.api.types.is_numeric_dtype(pbp[c]) and not any(k in c for k in DROP_SUBSTR)]


def pair_features(cf, prows, names, cat_maps):
    """(n_lines * n_runs, len(names)) float matrix, rows line-major. cf: the matcher's chart features (one row per
    line); prows: the candidate runs' pbp rows; names: feature names ('c_<chart feature>', a pbp column, or a 'd_'
    difference); cat_maps: {column: [categories]} - a categorical column becomes its category's index."""
    n, m = len(cf), len(prows)
    li, ri = np.repeat(np.arange(n), m), np.tile(np.arange(m), n)
    col = {}

    def chart(c):
        return cf[c].to_numpy(float)[li] if c in cf else np.full(n * m, np.nan)

    def play(c):
        if c not in prows:
            return np.full(n * m, np.nan)
        v = prows[c].to_numpy()
        if c in cat_maps:
            idx = {k: i for i, k in enumerate(cat_maps[c])}
            v = np.array([idx.get(str(x), np.nan) if x is not None and x == x else np.nan for x in v], float)
        else:
            v = pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(float)
        return v[ri]

    out = np.empty((n * m, len(names)))
    for j, name in enumerate(names):
        if name.startswith("c_"):
            out[:, j] = chart(name[2:])
        elif name == "d_gain":
            out[:, j] = chart("gain") - play("yards_gained")
        elif name == "d_len":
            out[:, j] = chart("path_len") - play("ngs_carrier_dist")
        elif name == "d_x0_hash":
            h = prows["starting_hash"].map(_HASH).to_numpy(float)[ri] if "starting_hash" in prows else np.nan
            out[:, j] = chart("x0") - h
        elif name == "d_end_hash_next":
            h = prows["next_hash_carry"].map(_SIDE).to_numpy(float)[ri] if "next_hash_carry" in prows else np.nan
            out[:, j] = chart("loc_x") * h
        else:
            out[:, j] = play(name)
    return out


def _load(path=None):
    global _MODEL
    if path is None and _MODEL is not None:
        return _MODEL
    p = path or MODEL_PATH
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    m = {k: z[k] for k in z.files}
    meta = json.loads(str(m.pop("meta")))
    m["names"], m["cat_maps"], m["pbp_cols"] = meta["names"], meta["cat_maps"], meta["pbp_cols"]
    if path is None:
        _MODEL = m
    return m


def predict_logit(X, model):
    """The ensemble's raw score per row (log odds that the pair is the line's run)."""
    out = np.full(len(X), float(model["baseline"]))
    rows = np.arange(len(X))
    for t in range(len(model["tree_start"])):
        base = int(model["tree_start"][t])
        node = np.zeros(len(X), int)
        for _ in range(int(model["max_depth"]) + 1):
            g = base + node
            leaf = model["is_leaf"][g].astype(bool)
            if leaf.all():
                break
            x = X[rows, model["feature"][g]]
            left = np.where(np.isnan(x), model["missing_left"][g].astype(bool), x <= model["threshold"][g])
            node = np.where(leaf, node, np.where(left, model["left"][g], model["right"][g]))
        out += model["value"][base + node]
    return out


def pair_cost(cf, prows, model=None):
    """(n_lines, n_runs) -log odds of each line being each run, centred per line (0 for no model)."""
    model = model or _load()
    if model is None or not len(cf) or not len(prows):
        return np.zeros((len(cf), len(prows)))
    X = pair_features(cf, prows, model["names"], model["cat_maps"])
    z = predict_logit(X, model).reshape(len(cf), len(prows))
    cost = -z
    return cost - cost.mean(axis=1, keepdims=True)


def export(clf, names, cat_maps, pbp_cols, path=MODEL_PATH):
    """Write a fitted sklearn HistGradientBoostingClassifier (numeric splits only) as plain arrays."""
    trees = [p[0].nodes for p in clf._predictors]
    starts = np.cumsum([0] + [len(t) for t in trees[:-1]])
    allnodes = np.concatenate(trees)
    depth = max(int(t["depth"].max()) for t in trees)
    meta = json.dumps(dict(names=names, cat_maps=cat_maps, pbp_cols=pbp_cols))
    np.savez_compressed(path, baseline=np.float64(np.ravel(clf._baseline_prediction)[0]), tree_start=starts.astype(np.int64),
                        feature=allnodes["feature_idx"].astype(np.int64), threshold=allnodes["num_threshold"].astype(float),
                        missing_left=allnodes["missing_go_to_left"].astype(np.int8), left=allnodes["left"].astype(np.int64),
                        right=allnodes["right"].astype(np.int64), is_leaf=allnodes["is_leaf"].astype(np.int8),
                        value=allnodes["value"].astype(float), max_depth=np.int64(depth), meta=np.array(meta))

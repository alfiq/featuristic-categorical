"""Categorical interaction features generated in Python before the Nim GA.

The native GA only sees a numeric matrix, so it cannot express "x minus the
mean of x within its category" or "the effect of the (a, b) category pair".
This module derives such columns up front so the GA can use them as leaves:

* per categorical column: frequency encoding
* per (categorical, numeric) pair: group mean, group difference
  (``x - group_mean``), group ratio (``x / group_mean``) and group std
* per pair of categorical columns: cross-fitted pair target encoding and
  pair frequency

All statistics are learned on the training data and frozen for ``transform``.
Target-based columns are cross-fitted on the training rows (sklearn
``TargetEncoder.fit_transform``) so they do not leak ``y``. Categories unseen
at fit time fall back to the average training group value (or 0 frequency).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.multiclass import type_of_target

_EPS = 1e-9


def _as_key(s: pd.Series) -> pd.Series:
    """Stable string keys for a categorical column (same at fit/transform)."""
    if is_numeric_dtype(s) and not s.dtype == bool:
        s = pd.to_numeric(s, errors="coerce").astype(np.float64)
        return s.map(repr).astype(object)
    return s.astype(str).astype(object)


def _pair_key(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a + "\x1f" + b).astype(object)


def _score(col: np.ndarray, y: np.ndarray) -> float:
    """Cheap relevance score used only to rank candidates when capping.

    max(|corr(f, y)|, |corr(|f - median(f)|, y)|) so that features whose
    effect is symmetric (e.g. absolute deviation from a group mean) are not
    discarded for having zero linear correlation.
    """
    best = 0.0
    for f in (col, np.abs(col - np.median(col))):
        sd = f.std()
        if sd < _EPS or not np.isfinite(sd):
            continue
        c = np.corrcoef(f, y)[0, 1]
        if np.isfinite(c):
            best = max(best, abs(float(c)))
    return best


class CategoricalFeatureGenerator:
    """Stateful generator of categorical interaction columns.

    Parameters
    ----------
    max_features : int
        Maximum number of generated columns. When more candidates exist,
        the highest-ranked ones (by a cheap univariate score computed on the
        training data, using cross-fitted values for target encodings) are kept.
    max_cardinality : int
        Numeric columns with at most this many distinct values are also treated
        as categorical. Non-numeric columns are always categorical. Numeric
        columns with more distinct values are never used as group keys.
    cv : int
        Folds for cross-fitting the pair target encoding.
    random_state : int or None
        Seed for the cross-fitting split.
    """

    def __init__(self, max_features=50, max_cardinality=50, cv=5, random_state=None):
        self.max_features = max_features
        self.max_cardinality = max_cardinality
        self.cv = cv
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        X = X.reset_index(drop=True)
        y_arr = np.asarray(y, dtype=np.float64)
        n = len(X)

        self.cat_cols_ = []
        self.num_cols_ = []
        for c in X.columns:
            s = X[c]
            k = s.nunique()
            if not is_numeric_dtype(s):
                # Skip ID-like columns where every row is its own category.
                if 2 <= k < n:
                    self.cat_cols_.append(c)
            else:
                # Low-cardinality numerics are grouped too, but only when the
                # groups are populated enough (>= 5 rows each on average).
                if 2 <= k <= self.max_cardinality and n >= 5 * k:
                    self.cat_cols_.append(c)
                if k > 2:
                    self.num_cols_.append(c)

        kind = type_of_target(y_arr)
        self.target_type_ = "binary" if kind == "binary" else "continuous"

        keys = {c: _as_key(X[c]) for c in self.cat_cols_}
        candidates = []  # (name, spec)
        for c in self.cat_cols_:
            candidates.append((f"{c}__freq", ("freq", c)))
        for c in self.cat_cols_:
            for x in self.num_cols_:
                if x == c:
                    continue
                for op in ("gdiff", "gmean", "gratio", "gstd"):
                    candidates.append((f"{c}__{op}__{x}", (op, c, x)))
        for a, b in combinations(self.cat_cols_, 2):
            candidates.append((f"{a}__x__{b}__te", ("pair_te", a, b)))
            candidates.append((f"{a}__x__{b}__freq", ("pair_freq", a, b)))

        # Learn the state for every candidate and produce training values.
        self.state_ = {}
        values = {}
        taken = {str(c) for c in X.columns}
        for name, spec in candidates:
            if name in taken:
                continue
            vals, st = self._fit_one(spec, X, keys, y_arr)
            if vals is None:
                continue
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            if vals.std() < _EPS:
                continue
            self.state_[name] = (spec, st)
            values[name] = vals

        names = list(values)
        if self.max_features is not None and len(names) > self.max_features:
            scores = {nm: _score(values[nm], y_arr) for nm in names}
            keep = set(sorted(names, key=lambda nm: -scores[nm])[: self.max_features])
            names = [nm for nm in names if nm in keep]
            self.state_ = {nm: self.state_[nm] for nm in names}

        self.feature_names_out_ = names
        return pd.DataFrame({nm: values[nm] for nm in names}, index=X.index)

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def _fit_one(self, spec, X, keys, y):
        op = spec[0]
        if op == "freq":
            k = keys[spec[1]]
            freq = k.value_counts(normalize=True)
            return k.map(freq).to_numpy(np.float64), {"map": freq.to_dict()}
        if op in ("gdiff", "gmean", "gratio", "gstd"):
            k, x = keys[spec[1]], X[spec[2]].astype(np.float64)
            grp = x.groupby(k)
            if op == "gstd":
                stat = grp.std(ddof=0).fillna(0.0)
            else:
                stat = grp.mean()
            fallback = float(stat.mean())
            g = k.map(stat).to_numpy(np.float64)
            st = {"map": stat.to_dict(), "fallback": fallback}
            return self._apply_group(op, x.to_numpy(np.float64), g), st
        if op == "pair_freq":
            k = _pair_key(keys[spec[1]], keys[spec[2]])
            freq = k.value_counts(normalize=True)
            return k.map(freq).to_numpy(np.float64), {"map": freq.to_dict()}
        if op == "pair_te":
            if len(X) < 2 * self.cv:
                return None, None  # too few rows to cross-fit safely
            k = _pair_key(keys[spec[1]], keys[spec[2]])
            enc = TargetEncoder(
                target_type=self.target_type_,
                cv=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
            # fit_transform is cross-fitted: each training row is encoded by a
            # model that never saw its own target.
            vals = enc.fit_transform(k.to_frame(), y).ravel()
            return vals.astype(np.float64), {"encoder": enc}
        raise ValueError(op)

    @staticmethod
    def _apply_group(op, x, g):
        if op == "gmean" or op == "gstd":
            return g
        if op == "gdiff":
            return x - g
        # gratio
        out = np.zeros_like(x)
        ok = np.abs(g) > _EPS
        out[ok] = x[ok] / g[ok]
        return out

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reset_index(drop=True)
        keys = {}

        def key(c):
            if c not in keys:
                keys[c] = _as_key(X[c])
            return keys[c]

        out = {}
        for name in self.feature_names_out_:
            spec, st = self.state_[name]
            op = spec[0]
            if op == "freq":
                v = key(spec[1]).map(st["map"]).fillna(0.0).to_numpy(np.float64)
            elif op == "pair_freq":
                k = _pair_key(key(spec[1]), key(spec[2]))
                v = k.map(st["map"]).fillna(0.0).to_numpy(np.float64)
            elif op == "pair_te":
                k = _pair_key(key(spec[1]), key(spec[2]))
                v = st["encoder"].transform(k.to_frame()).ravel().astype(np.float64)
            else:
                g = (
                    key(spec[1])
                    .map(st["map"])
                    .astype(np.float64)
                    .fillna(st["fallback"])
                    .to_numpy(np.float64)
                )
                x = X[spec[2]].astype(np.float64).to_numpy()
                v = self._apply_group(op, x, g)
            out[name] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(out, index=X.index, columns=self.feature_names_out_)

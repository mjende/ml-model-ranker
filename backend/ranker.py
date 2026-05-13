"""Ranking logic: normalization, weighted scoring, justification."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS: Dict[str, float] = {
    "popularity": 0.20,
    "architecture": 0.10,
    "size": 0.15,
    "accuracy": 0.30,
    "speed": 0.15,
    "documentation": 0.10,
}

# Architecture preference scores (typical -> 0.5, unique/desired -> 1.0)
ARCH_PREFERENCE: Dict[str, float] = {
    "transformer": 0.8,
    "llm": 0.9,
    "moe": 1.0,
    "mixture-of-experts": 1.0,
    "mamba": 1.0,
    "ssm": 1.0,
    "diffusion": 0.9,
    "cnn": 0.5,
    "rnn": 0.4,
    "lstm": 0.4,
    "mlp": 0.3,
    "hybrid": 0.9,
    "multimodal": 1.0,
    "vit": 0.8,
}


def _log_minmax(series: pd.Series) -> pd.Series:
    """Logarithmic + min-max normalization to [0, 1]."""
    s = series.astype(float)
    mask = s.notna()
    if mask.sum() == 0:
        return s
    logged = np.log1p(s[mask].clip(lower=0))
    lo, hi = logged.min(), logged.max()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if hi - lo < 1e-9:
        out[mask] = 0.5
    else:
        out[mask] = (logged - lo) / (hi - lo)
    return out


def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mask = s.notna()
    if mask.sum() == 0:
        return s
    lo, hi = s[mask].min(), s[mask].max()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if hi - lo < 1e-9:
        out[mask] = 0.5
    else:
        out[mask] = (s[mask] - lo) / (hi - lo)
    return out


def _inverted_log_minmax(series: pd.Series) -> pd.Series:
    """Smaller = better. Useful for size, latency."""
    norm = _log_minmax(series)
    return 1.0 - norm


def _inverted_minmax(series: pd.Series) -> pd.Series:
    norm = _minmax(series)
    return 1.0 - norm


def architecture_score(arch: Any) -> float:
    if not isinstance(arch, str) or not arch.strip():
        return float("nan")
    key = arch.strip().lower()
    # Direct hit
    if key in ARCH_PREFERENCE:
        return ARCH_PREFERENCE[key]
    # Substring match
    for k, v in ARCH_PREFERENCE.items():
        if k in key:
            return v
    return 0.5  # unknown but provided -> neutral-positive


def documentation_score(value: Any) -> float:
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        v = float(value)
        if v > 1.0:
            v = v / 100.0
        return max(0.0, min(1.0, v))
    if isinstance(value, str):
        v = value.strip().lower()
        mapping = {"full": 1.0, "high": 1.0, "good": 0.8, "partial": 0.5,
                   "medium": 0.5, "low": 0.3, "none": 0.0, "missing": 0.0}
        if v in mapping:
            return mapping[v]
    return float("nan")


@dataclass
class RankingResult:
    df: pd.DataFrame
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


def compute_ranking(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> RankingResult:
    """Compute weighted ranking score for each model.

    Expected (case-insensitive) input columns include any subset of:
      - model / name           (required)
      - architecture
      - parameters / param_count / size
      - accuracy / quality / score_quality
      - downloads / popularity (e.g. HF downloads)
      - stars / github_stars
      - latency / speed_ms
      - documentation / doc_score
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    df = df.copy()

    # Normalize column names: lower-case map
    lower_map = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    if "model" not in df.columns:
        for alt in ("name", "model_name", "id", "repo"):
            if alt in df.columns:
                df["model"] = df[alt]
                break
    if "model" not in df.columns:
        raise ValueError("Input must contain a 'Model' / 'name' column.")

    # Popularity: prefer combined downloads + stars (log-scaled, then averaged)
    pop_sources = []
    if "downloads" in df.columns:
        pop_sources.append(_log_minmax(pd.to_numeric(df["downloads"], errors="coerce")))
    if "stars" in df.columns:
        pop_sources.append(_log_minmax(pd.to_numeric(df["stars"], errors="coerce")))
    if "popularity" in df.columns and not pop_sources:
        pop_sources.append(_log_minmax(pd.to_numeric(df["popularity"], errors="coerce")))
    if pop_sources:
        pop_norm = pd.concat(pop_sources, axis=1).mean(axis=1, skipna=True)
    else:
        pop_norm = pd.Series([float("nan")] * len(df), index=df.index)

    # Architecture
    if "architecture" in df.columns:
        arch_norm = df["architecture"].apply(architecture_score)
    else:
        arch_norm = pd.Series([float("nan")] * len(df), index=df.index)

    # Size: prefer parameters/param_count; smaller -> better
    size_col = None
    for c in ("parameters", "param_count", "params", "size"):
        if c in df.columns:
            size_col = c
            break
    if size_col is not None:
        size_norm = _inverted_log_minmax(pd.to_numeric(df[size_col], errors="coerce"))
    else:
        size_norm = pd.Series([float("nan")] * len(df), index=df.index)

    # Accuracy / quality (higher -> better)
    acc_col = None
    for c in ("accuracy", "quality", "score_quality", "f1", "bleu"):
        if c in df.columns:
            acc_col = c
            break
    if acc_col is not None:
        acc_series = pd.to_numeric(df[acc_col], errors="coerce")
        # Heuristic: if values look like percentages, scale to 0-1
        if acc_series.dropna().gt(1.5).any():
            acc_series = acc_series / 100.0
        acc_norm = _minmax(acc_series)
    else:
        acc_norm = pd.Series([float("nan")] * len(df), index=df.index)

    # Speed / latency (smaller latency -> better; if "throughput" then higher -> better)
    speed_norm = pd.Series([float("nan")] * len(df), index=df.index)
    if "latency" in df.columns or "speed_ms" in df.columns:
        col = "latency" if "latency" in df.columns else "speed_ms"
        speed_norm = _inverted_log_minmax(pd.to_numeric(df[col], errors="coerce"))
    elif "throughput" in df.columns:
        speed_norm = _log_minmax(pd.to_numeric(df["throughput"], errors="coerce"))
    elif "speed" in df.columns:
        s = pd.to_numeric(df["speed"], errors="coerce")
        speed_norm = _log_minmax(s)

    # Documentation
    doc_col = None
    for c in ("documentation", "doc_score", "docs"):
        if c in df.columns:
            doc_col = c
            break
    if doc_col is not None:
        doc_norm = df[doc_col].apply(documentation_score)
    else:
        doc_norm = pd.Series([float("nan")] * len(df), index=df.index)

    norms = pd.DataFrame({
        "popularity": pop_norm,
        "architecture": arch_norm,
        "size": size_norm,
        "accuracy": acc_norm,
        "speed": speed_norm,
        "documentation": doc_norm,
    })

    # Missing data handling: neutral imputation (mean across column) AND rescale weights
    def _row_score(row: pd.Series) -> float:
        present = {k: v for k, v in row.items() if pd.notna(v)}
        if not present:
            return 0.0
        total_w = sum(w[k] for k in present)
        if total_w <= 0:
            return 0.0
        return sum((w[k] / total_w) * v for k, v in present.items())

    scores = norms.apply(_row_score, axis=1)
    df["score"] = (scores * 100).round(2)

    # Attach normalized columns for transparency
    for c in norms.columns:
        df[f"{c}_norm"] = norms[c].round(4)

    # Justification text per row
    df["justification"] = norms.apply(lambda r: _justify(r, w), axis=1)

    df.sort_values("score", ascending=False, inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "rank", df.index + 1)

    return RankingResult(df=df, weights=w)


def _justify(row: pd.Series, weights: Dict[str, float]) -> str:
    """Generate a short Polish justification based on dominant factors."""
    contribs = {k: (weights.get(k, 0) * v) for k, v in row.items() if pd.notna(v)}
    if not contribs:
        return "Brak wystarczających danych – wynik neutralny."
    sorted_c = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
    top = sorted_c[:2]
    weak = [k for k, v in sorted_c if v < 0.3 * weights.get(k, 1)][:1]

    labels = {
        "popularity": "popularność",
        "architecture": "architektura",
        "size": "kompaktowy rozmiar",
        "accuracy": "jakość/precyzja",
        "speed": "szybkość inferencji",
        "documentation": "dokumentacja/wsparcie",
    }

    parts = []
    if top:
        parts.append("Mocne strony: " + ", ".join(labels[k] for k, _ in top) + ".")
    if weak:
        parts.append("Słabe: " + ", ".join(labels[k] for k in weak) + ".")
    missing = [k for k in weights if k not in row.index or pd.isna(row.get(k))]
    if missing:
        parts.append("Brak danych: " + ", ".join(labels[k] for k in missing) + ".")
    return " ".join(parts)

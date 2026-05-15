"""Ranking logic: normalization, weighted scoring, justification.

Criteria supported:
    popularity      - HF downloads, HF likes, GitHub stars (log-normalized, averaged)
    architecture    - preference score by family + diversity bonus (rare arch -> +)
    size            - inverse of parameter count (smaller -> higher)
    accuracy        - quality metric (accuracy / F1 / BLEU / score)
    speed           - inverse of latency (or throughput if provided)
    documentation   - documentation completeness 0-1
    recency         - freshness from last_modified (newer -> higher; half-life decay)
    license         - permissiveness (Apache/MIT > custom > non-commercial/research)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS: Dict[str, float] = {
    "popularity": 0.18,
    "architecture": 0.10,
    "size": 0.12,
    "accuracy": 0.25,
    "speed": 0.12,
    "documentation": 0.08,
    "recency": 0.08,
    "license": 0.07,
}

# Built-in weight presets users can pick from the UI.
WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "balanced": dict(DEFAULT_WEIGHTS),
    "qa_focus": {  # exhaustive QA: quality + docs matter most
        "popularity": 0.10, "architecture": 0.10, "size": 0.10, "accuracy": 0.30,
        "speed": 0.10, "documentation": 0.15, "recency": 0.05, "license": 0.10,
    },
    "performance": {  # constrained hardware: small & fast wins
        "popularity": 0.10, "architecture": 0.10, "size": 0.25, "accuracy": 0.15,
        "speed": 0.25, "documentation": 0.05, "recency": 0.05, "license": 0.05,
    },
    "research": {  # newest unique architectures, ignore commercial license concerns
        "popularity": 0.10, "architecture": 0.25, "size": 0.05, "accuracy": 0.20,
        "speed": 0.05, "documentation": 0.05, "recency": 0.25, "license": 0.05,
    },
    "production": {  # battle-tested: popularity, license, docs > everything
        "popularity": 0.25, "architecture": 0.05, "size": 0.10, "accuracy": 0.20,
        "speed": 0.10, "documentation": 0.15, "recency": 0.05, "license": 0.10,
    },
}

# Architecture preference scores
ARCH_PREFERENCE: Dict[str, float] = {
    "transformer": 0.8, "llm": 0.9, "moe": 1.0, "mixture-of-experts": 1.0,
    "mamba": 1.0, "ssm": 1.0, "diffusion": 0.9, "cnn": 0.5, "rnn": 0.4,
    "lstm": 0.4, "mlp": 0.3, "hybrid": 0.9, "multimodal": 1.0, "vit": 0.8,
    "ocr": 0.85, "vision": 0.7, "embedding": 0.6,
}

# License permissiveness mapping (substrings of lowercased license).
# 1.0 = permissive (production-safe), 0.0 = research-only / non-commercial.
LICENSE_SCORE: Dict[str, float] = {
    "apache": 1.0, "apache-2.0": 1.0, "apache_2_0": 1.0,
    "mit": 1.0, "bsd": 1.0, "bsd-3": 1.0, "bsd-2": 1.0,
    "isc": 1.0, "unlicense": 1.0, "openrail": 0.85, "bigscience-openrail": 0.85,
    "creativeml-openrail": 0.8, "rail": 0.75,
    "cc-by-4": 0.9, "cc-by-sa": 0.8, "cc0": 1.0,
    "llama": 0.6, "llama2": 0.6, "llama3": 0.6,
    "gemma": 0.6, "mistral": 0.7,
    "apple-ascl": 0.5, "qwen": 0.65,
    "cc-by-nc": 0.2, "noncommercial": 0.2, "non-commercial": 0.2,
    "research": 0.15, "academic": 0.2, "research-only": 0.1,
    "other": 0.5, "unknown": float("nan"),
}


# ---------- normalization helpers ----------

def _log_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mask = s.notna()
    if mask.sum() == 0:
        return s.astype(float)
    logged = np.log1p(s[mask].clip(lower=0))
    lo, hi = logged.min(), logged.max()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if hi - lo < 1e-9:
        out[mask] = 0.5
    else:
        out[mask] = (logged - lo) / (hi - lo)
    return out


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mask = s.notna()
    if mask.sum() == 0:
        return s.astype(float)
    lo, hi = s[mask].min(), s[mask].max()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if hi - lo < 1e-9:
        out[mask] = 0.5
    else:
        out[mask] = (s[mask] - lo) / (hi - lo)
    return out


def _inverted_log_minmax(series: pd.Series) -> pd.Series:
    return 1.0 - _log_minmax(series)


# ---------- per-criterion scorers ----------

def architecture_score(arch: Any) -> float:
    if not isinstance(arch, str) or not arch.strip():
        return float("nan")
    key = arch.strip().lower()
    if key in ARCH_PREFERENCE:
        return ARCH_PREFERENCE[key]
    for k, v in ARCH_PREFERENCE.items():
        if k in key:
            return v
    return 0.5


def documentation_score(value: Any) -> float:
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        v = float(value)
        if v > 1.0:
            v /= 100.0
        return max(0.0, min(1.0, v))
    if isinstance(value, str):
        v = value.strip().lower()
        mapping = {"full": 1.0, "high": 1.0, "good": 0.8, "partial": 0.5,
                   "medium": 0.5, "low": 0.3, "none": 0.0, "missing": 0.0}
        if v in mapping:
            return mapping[v]
    return float("nan")


def license_score(value: Any) -> float:
    if not isinstance(value, str):
        return float("nan")
    v = value.strip().lower().replace(" ", "-")
    if not v or v in ("nan",):
        return float("nan")
    # exact match
    if v in LICENSE_SCORE:
        return LICENSE_SCORE[v]
    # substring match (apache-2.0, apache_2_0, etc.)
    for k, s in LICENSE_SCORE.items():
        if k in v:
            return s
    return 0.5


def recency_score(value: Any, half_life_days: float = 365.0) -> float:
    """Exponential decay from `last_modified`. 0 days old -> 1.0, half-life -> 0.5."""
    if not value:
        return float("nan")
    if isinstance(value, float) and math.isnan(value):
        return float("nan")
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return float("nan")
        if isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
        else:
            dt = ts  # type: ignore
    except Exception:
        return float("nan")
    now = datetime.now(timezone.utc)
    age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return float(math.exp(-math.log(2) * age_days / half_life_days))


def _diversity_bonus(arch_series: pd.Series) -> pd.Series:
    """+ bonus for rare architectures (encourages variety in QA selection)."""
    out = pd.Series(np.nan, index=arch_series.index, dtype=float)
    keys = arch_series.astype(str).str.lower().str.strip()
    counts = keys.value_counts(dropna=False)
    total = max(1, len(arch_series))
    for idx in arch_series.index:
        k = keys.loc[idx]
        if not k or k in ("nan", "none", ""):
            continue
        # Rarer => higher; clamp to [0, 1].
        out.loc[idx] = max(0.0, min(1.0, 1.0 - (counts.get(k, 1) - 1) / total))
    return out


# ---------- main entrypoint ----------

@dataclass
class RankingResult:
    df: pd.DataFrame
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={c: re.sub(r"[\s_]+", "_", str(c).strip().lower()) for c in df.columns},
              inplace=True)
    return df


def _ensure_model_column(df: pd.DataFrame) -> None:
    if "model" in df.columns:
        return
    for alt in ("model_name", "name", "id", "repo", "hf_id"):
        if alt in df.columns:
            df["model"] = df[alt]
            return
    raise ValueError("Input must contain a 'Model' / 'name' column.")


def _pick(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def compute_ranking(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> RankingResult:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    df = _normalize_columns(df)
    _ensure_model_column(df)
    n = len(df)
    nan_series = pd.Series([np.nan] * n, index=df.index, dtype=float)

    # --- popularity: combine downloads + likes + stars ---
    pop_sources = []
    for col in ("downloads", "likes", "stars"):
        if col in df.columns:
            pop_sources.append(_log_minmax(df[col]))
    if not pop_sources and "popularity" in df.columns:
        pop_sources.append(_log_minmax(df["popularity"]))
    pop_norm = (pd.concat(pop_sources, axis=1).mean(axis=1, skipna=True)
                if pop_sources else nan_series.copy())

    # --- architecture: family preference + diversity bonus ---
    arch_col = _pick(df, ["architecture", "architecture_detected", "model_type"])
    if arch_col:
        family = df[arch_col].apply(architecture_score)
        diversity = _diversity_bonus(df[arch_col])
        # weighted blend: 70% family score, 30% diversity bonus
        arch_norm = family.where(family.notna(), other=0.5) * 0.7 + diversity.fillna(0.0) * 0.3
        # If both are NaN we want NaN
        arch_norm = arch_norm.where(family.notna() | diversity.notna(), other=np.nan)
    else:
        arch_norm = nan_series.copy()

    # --- size ---
    size_col = _pick(df, ["parameters", "param_count", "params", "size"])
    size_norm = _inverted_log_minmax(df[size_col]) if size_col else nan_series.copy()

    # --- accuracy ---
    acc_col = _pick(df, ["accuracy", "quality", "score_quality", "f1", "bleu"])
    if acc_col:
        acc_series = pd.to_numeric(df[acc_col], errors="coerce")
        if acc_series.dropna().gt(1.5).any():
            acc_series = acc_series / 100.0
        acc_norm = _minmax(acc_series)
    else:
        acc_norm = nan_series.copy()

    # --- speed ---
    if "latency" in df.columns or "speed_ms" in df.columns:
        col = "latency" if "latency" in df.columns else "speed_ms"
        speed_norm = _inverted_log_minmax(df[col])
    elif "throughput" in df.columns:
        speed_norm = _log_minmax(df["throughput"])
    elif "speed" in df.columns:
        speed_norm = _log_minmax(df["speed"])
    else:
        speed_norm = nan_series.copy()

    # --- documentation ---
    doc_col = _pick(df, ["documentation", "doc_score", "docs"])
    doc_norm = df[doc_col].apply(documentation_score) if doc_col else nan_series.copy()

    # --- recency ---
    recency_col = _pick(df, ["last_modified", "last_modified_at", "updated", "lastmodified"])
    recency_norm = df[recency_col].apply(recency_score) if recency_col else nan_series.copy()

    # --- license ---
    license_col = _pick(df, ["license", "licence"])
    license_norm = df[license_col].apply(license_score) if license_col else nan_series.copy()

    norms = pd.DataFrame({
        "popularity": pop_norm,
        "architecture": arch_norm,
        "size": size_norm,
        "accuracy": acc_norm,
        "speed": speed_norm,
        "documentation": doc_norm,
        "recency": recency_norm,
        "license": license_norm,
    })

    # Score = weighted average with renormalization when some criteria are missing.
    def _row_score(row: pd.Series) -> float:
        present = {k: v for k, v in row.items() if pd.notna(v)}
        if not present:
            return 0.0
        total_w = sum(w.get(k, 0) for k in present)
        if total_w <= 0:
            return 0.0
        return sum((w.get(k, 0) / total_w) * v for k, v in present.items())

    df["score"] = (norms.apply(_row_score, axis=1) * 100).round(2)

    # Coverage: how many criteria with positive weight were evaluable.
    weighted_keys = [k for k, v in w.items() if v > 0]
    if weighted_keys:
        df["coverage"] = (
            norms[weighted_keys].notna().sum(axis=1) / len(weighted_keys)
        ).round(2)
    else:
        df["coverage"] = 0.0

    # Confidence-adjusted score: penalize models evaluated on very few criteria.
    # A model with 100% coverage keeps its score; with 0% coverage drops to 60%.
    df["score_adj"] = (df["score"] * (0.6 + 0.4 * df["coverage"])).round(2)

    for c in norms.columns:
        df[f"{c}_norm"] = norms[c].round(4)

    raw = pd.DataFrame({
        "downloads": pd.to_numeric(df.get("downloads"), errors="coerce") if "downloads" in df.columns else nan_series,
        "likes": pd.to_numeric(df.get("likes"), errors="coerce") if "likes" in df.columns else nan_series,
        "stars": pd.to_numeric(df.get("stars"), errors="coerce") if "stars" in df.columns else nan_series,
        "parameters": pd.to_numeric(df.get(size_col), errors="coerce") if size_col else nan_series,
        "accuracy": pd.to_numeric(df.get(acc_col), errors="coerce") if acc_col else nan_series,
        "latency": pd.to_numeric(df.get("latency"), errors="coerce") if "latency" in df.columns else nan_series,
        "architecture": df.get(arch_col) if arch_col else pd.Series([None] * n, index=df.index),
        "license": df.get(license_col) if license_col else pd.Series([None] * n, index=df.index),
        "last_modified": df.get(recency_col) if recency_col else pd.Series([None] * n, index=df.index),
    })
    df["justification"] = [_justify(norms.iloc[i], raw.iloc[i], w) for i in range(n)]

    df.sort_values(["score_adj", "coverage", "score"], ascending=[False, False, False],
                   inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "rank", df.index + 1)

    # Rename adjusted to be the primary 'score' shown; keep raw as score_raw.
    df.rename(columns={"score": "score_raw", "score_adj": "score"}, inplace=True)

    return RankingResult(df=df, weights=w)


# ---------- justification rendering ----------

def _fmt_num(v: Any) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "?"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    av = abs(v)
    if av >= 1e9:
        return f"{v/1e9:.1f}B"
    if av >= 1e6:
        return f"{v/1e6:.1f}M"
    if av >= 1e3:
        return f"{v/1e3:.1f}k"
    if not float(v).is_integer():
        return f"{v:.2f}"
    return str(int(v))


def _fmt_age(value: Any) -> Optional[str]:
    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        age = (datetime.now(timezone.utc) - ts.to_pydatetime()).days
    except Exception:
        return None
    if age < 30:
        return f"świeży ({age} dni)"
    if age < 365:
        return f"{age // 30} mies."
    return f"{age // 365} lat"


LABELS = {
    "popularity": "popularność",
    "architecture": "architektura",
    "size": "kompaktowy rozmiar",
    "accuracy": "jakość",
    "speed": "szybkość",
    "documentation": "dokumentacja",
    "recency": "świeżość",
    "license": "licencja",
}


def _justify(norm_row: pd.Series, raw_row: pd.Series, weights: Dict[str, float]) -> str:
    facts: List[str] = []
    if pd.notna(raw_row.get("downloads")):
        facts.append(f"pobrania HF: {_fmt_num(raw_row['downloads'])}")
    if pd.notna(raw_row.get("likes")):
        facts.append(f"♥ HF: {_fmt_num(raw_row['likes'])}")
    if pd.notna(raw_row.get("stars")):
        facts.append(f"⭐ GH: {_fmt_num(raw_row['stars'])}")
    if pd.notna(raw_row.get("parameters")):
        facts.append(f"params: {_fmt_num(raw_row['parameters'])}")
    if pd.notna(raw_row.get("accuracy")):
        v = raw_row["accuracy"]
        facts.append(f"jakość: {v*100:.1f}%" if 0 <= v <= 1.5 else f"jakość: {v:.1f}")
    if pd.notna(raw_row.get("latency")):
        facts.append(f"latencja: {_fmt_num(raw_row['latency'])} ms")
    arch = raw_row.get("architecture")
    if isinstance(arch, str) and arch.strip():
        facts.append(f"arch: {arch.strip()}")
    lic = raw_row.get("license")
    if isinstance(lic, str) and lic.strip():
        facts.append(f"licencja: {lic.strip()}")
    age = _fmt_age(raw_row.get("last_modified"))
    if age:
        facts.append(f"wiek: {age}")

    contribs = {k: weights.get(k, 0) * v for k, v in norm_row.items() if pd.notna(v)}
    parts: List[str] = []
    if facts:
        parts.append(" · ".join(facts))
    if contribs:
        sorted_c = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
        strong = [LABELS[k] for k, _ in sorted_c if norm_row.get(k, 0) >= 0.7][:3]
        weak = [LABELS[k] for k, _ in sorted_c
                if norm_row.get(k, 1) <= 0.3 and weights.get(k, 0) > 0][:2]
        if strong:
            parts.append("Mocne: " + ", ".join(strong) + ".")
        if weak:
            parts.append("Słabe: " + ", ".join(weak) + ".")
    missing = [LABELS[k] for k in weights
               if (k not in norm_row.index or pd.isna(norm_row.get(k)))
               and weights.get(k, 0) > 0]
    if missing:
        parts.append("Brak danych: " + ", ".join(missing) + ".")
    return " | ".join(parts) if parts else "Brak wystarczających danych – wynik neutralny."

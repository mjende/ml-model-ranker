"""Optional enrichment from Hugging Face and GitHub public APIs.

Network calls have short timeouts and degrade gracefully on errors.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

HF_API = "https://huggingface.co/api/models/"
GH_API = "https://api.github.com/repos/"
TIMEOUT = 8.0

log = logging.getLogger("ml-ranker.enrichment")


def _session() -> requests.Session:
    """Build a requests Session that respects HTTPS_PROXY / HTTP_PROXY env vars."""
    s = requests.Session()
    s.trust_env = True  # honor HTTPS_PROXY env var
    return s


def _hf_lookup(model_id: str, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    s = session or _session()
    try:
        r = s.get(HF_API + model_id, timeout=TIMEOUT)
        if r.status_code != 200:
            log.info("HF lookup %s -> HTTP %s", model_id, r.status_code)
            return {}
        data = r.json()
        out: Dict[str, Any] = {
            "downloads": data.get("downloads"),
            "likes": data.get("likes"),
            "hf_id": model_id,
        }
        tags = data.get("tags") or []
        if isinstance(tags, list) and tags:
            out["tags"] = ",".join(tags[:10])
        if data.get("pipeline_tag"):
            out["pipeline_tag"] = data["pipeline_tag"]
        if data.get("library_name"):
            out["library"] = data["library_name"]
        cfg = data.get("config") or {}
        if isinstance(cfg, dict):
            archs = cfg.get("architectures")
            if isinstance(archs, list) and archs:
                out["architecture_detected"] = archs[0]
            mt = cfg.get("model_type")
            if mt:
                out["model_type"] = mt
        safetensors = data.get("safetensors") or {}
        if isinstance(safetensors, dict):
            total = safetensors.get("total")
            if isinstance(total, (int, float)):
                out["parameters"] = int(total)
        # License: prefer cardData.license, fallback to top-level
        card = data.get("cardData") or {}
        if isinstance(card, dict) and card.get("license"):
            out["license"] = str(card["license"])
        elif data.get("license"):
            out["license"] = str(data["license"])
        if data.get("lastModified"):
            out["last_modified"] = data["lastModified"]
        return out
    except requests.RequestException as exc:
        log.warning("HF lookup failed for %s: %s", model_id, exc)
        return {}


_GH_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def _gh_lookup(repo: str, token: Optional[str] = None,
               session: Optional[requests.Session] = None) -> Dict[str, Any]:
    if not _GH_RE.match(repo):
        return {}
    s = session or _session()
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = s.get(GH_API + repo, headers=headers, timeout=TIMEOUT)
        if r.status_code != 200:
            return {}
        data = r.json()
        return {
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "open_issues": data.get("open_issues_count"),
        }
    except requests.RequestException as exc:
        log.warning("GH lookup failed for %s: %s", repo, exc)
        return {}


# Keys (case-insensitive, ignoring spaces/underscores) that may hold a model ID
_MODEL_ID_KEYS = {
    "model", "modelname", "model_id", "modelid", "name", "id",
    "repo", "repository", "huggingface", "hfid", "hfmodel",
    "examplehfmodels", "examplehfmodel",
}


def _normalize_key(k: str) -> str:
    return re.sub(r"[\s_]+", "", str(k).strip().lower())


def _extract_model_id(row: Dict[str, Any]) -> str:
    """Find a Hugging Face style 'org/name' identifier in the row, regardless of column name."""
    candidates: List[str] = []
    for k, v in row.items():
        if v is None or (isinstance(v, float) and v != v):
            continue
        norm_k = _normalize_key(k)
        if norm_k in _MODEL_ID_KEYS:
            text = str(v).strip()
            # "skt/A.X-K1, etc." -> first comma-separated piece
            first = re.split(r"[,;\n]", text, maxsplit=1)[0].strip()
            if first:
                candidates.append(first)
    # Prefer the first candidate that looks like a HF repo "org/name"
    for c in candidates:
        if "/" in c and not c.startswith(("http://", "https://")):
            return c
    return candidates[0] if candidates else ""


def enrich_row(row: Dict[str, Any], gh_token: Optional[str] = None,
               session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Fill missing fields for one model row using HF/GH lookups."""
    out = dict(row)
    s = session or _session()

    model_id = _extract_model_id(out)
    if not model_id:
        return out

    if "/" in model_id or re.match(r"^[\w.-]+$", model_id):
        hf = _hf_lookup(model_id, session=s)
        for k, v in hf.items():
            if v is None:
                continue
            if k not in out or out.get(k) in (None, "", float("nan")):
                out[k] = v

    gh_repo = out.get("github") or out.get("gh_repo")
    if not gh_repo and "/" in model_id and not model_id.startswith("http"):
        gh_repo = model_id
    if gh_repo:
        gh = _gh_lookup(str(gh_repo).strip(),
                        token=gh_token or os.getenv("GITHUB_TOKEN"),
                        session=s)
        for k, v in gh.items():
            if v is None:
                continue
            if k not in out or out.get(k) in (None, "", float("nan")):
                out[k] = v

    return out


def enrich_dataframe(df, gh_token: Optional[str] = None):
    import pandas as pd
    s = _session()
    rows = []
    hits = 0
    for _, row in df.iterrows():
        clean = {k: (None if (isinstance(v, float) and v != v) else v) for k, v in row.to_dict().items()}
        enriched = enrich_row(clean, gh_token=gh_token, session=s)
        if any(k in enriched and enriched[k] not in (None, "") for k in ("downloads", "parameters", "stars")):
            hits += 1
        rows.append(enriched)
    log.info("Enrichment hits: %d / %d", hits, len(rows))
    return pd.DataFrame(rows)

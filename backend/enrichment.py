"""Optional enrichment from Hugging Face and GitHub public APIs.

Network calls have short timeouts and degrade gracefully on errors.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import requests

HF_API = "https://huggingface.co/api/models/"
GH_API = "https://api.github.com/repos/"
TIMEOUT = 6.0


def _hf_lookup(model_id: str) -> Dict[str, Any]:
    try:
        r = requests.get(HF_API + model_id, timeout=TIMEOUT)
        if r.status_code != 200:
            return {}
        data = r.json()
        out: Dict[str, Any] = {
            "downloads": data.get("downloads"),
            "likes": data.get("likes"),
        }
        # Architecture from config / tags
        tags = data.get("tags") or []
        if isinstance(tags, list) and tags:
            out["tags"] = ",".join(tags[:10])
        cfg = data.get("config") or {}
        if isinstance(cfg, dict):
            archs = cfg.get("architectures")
            if isinstance(archs, list) and archs:
                out["architecture"] = archs[0]
        # Parameter count heuristic
        safetensors = data.get("safetensors") or {}
        if isinstance(safetensors, dict):
            total = safetensors.get("total")
            if isinstance(total, (int, float)):
                out["parameters"] = int(total)
        return out
    except requests.RequestException:
        return {}


_GH_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def _gh_lookup(repo: str, token: Optional[str] = None) -> Dict[str, Any]:
    if not _GH_RE.match(repo):
        return {}
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(GH_API + repo, headers=headers, timeout=TIMEOUT)
        if r.status_code != 200:
            return {}
        data = r.json()
        return {
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "open_issues": data.get("open_issues_count"),
        }
    except requests.RequestException:
        return {}


def enrich_row(row: Dict[str, Any], gh_token: Optional[str] = None) -> Dict[str, Any]:
    """Fill missing fields for one model row using HF/GH lookups."""
    out = dict(row)
    model_id = (out.get("model") or out.get("name") or out.get("id") or "").strip()
    if not model_id:
        return out

    # HF lookup if id looks like "org/name"
    if "/" in model_id:
        hf = _hf_lookup(model_id)
        for k, v in hf.items():
            if v is None:
                continue
            if k not in out or out.get(k) in (None, "", float("nan")):
                out[k] = v

    # GitHub: explicit "github" column, else try as repo
    gh_repo = out.get("github") or out.get("gh_repo")
    if not gh_repo and "/" in model_id and not model_id.startswith("http"):
        gh_repo = model_id
    if gh_repo:
        gh = _gh_lookup(str(gh_repo).strip(), token=gh_token or os.getenv("GITHUB_TOKEN"))
        for k, v in gh.items():
            if v is None:
                continue
            if k not in out or out.get(k) in (None, "", float("nan")):
                out[k] = v

    return out


def enrich_dataframe(df, gh_token: Optional[str] = None):
    rows = []
    for _, row in df.iterrows():
        clean = {k: (None if (isinstance(v, float) and v != v) else v) for k, v in row.to_dict().items()}
        rows.append(enrich_row(clean, gh_token=gh_token))
    import pandas as pd
    return pd.DataFrame(rows)

"""Flask backend for ML Model Ranker.

Run:
    python -m backend.app
or:
    flask --app backend.app run --port 8000
"""
from __future__ import annotations

import io
import json
import logging
import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

from .enrichment import enrich_dataframe
from .ranker import DEFAULT_WEIGHTS, WEIGHT_PRESETS, compute_ranking

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static",
)
# 16 MB upload limit
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def _json_error(message: str, status: int = 400) -> Response:
    resp = jsonify({"detail": message})
    resp.status_code = status
    return resp


def _parse_weights(weights_json: Optional[str]) -> Optional[Dict[str, float]]:
    if not weights_json:
        return None
    try:
        data = json.loads(weights_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Bad weights JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("weights must be an object")
    out: Dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid weight {k}={v}") from exc
    return out


_XLSX_MAGIC = b"PK\x03\x04"  # xlsx is a zip
_XLS_MAGIC = b"\xd0\xcf\x11\xe0"  # legacy xls (OLE2)


def _read_table(file_bytes: bytes, filename: str = "") -> pd.DataFrame:
    """Load a tabular file. Supports CSV ("," or ";") and Excel (.xlsx/.xls)."""
    name = (filename or "").lower()
    head = file_bytes[:4]
    is_excel = (
        name.endswith((".xlsx", ".xlsm", ".xls"))
        or head.startswith(_XLSX_MAGIC)
        or head.startswith(_XLS_MAGIC)
    )
    if is_excel:
        try:
            return pd.read_excel(io.BytesIO(file_bytes))
        except ImportError as exc:
            raise ValueError(
                "Brak biblioteki do XLSX (openpyxl). Zainstaluj: pip install openpyxl."
            ) from exc
        except Exception as exc:
            raise ValueError(f"Nie można odczytać pliku Excel: {exc}") from exc
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), sep=";")
        except Exception as exc:
            raise ValueError(f"Nie można sparsować pliku CSV/Excel: {exc}") from exc


def _get_form_bool(name: str) -> bool:
    val = request.form.get(name, "false").strip().lower()
    return val in {"1", "true", "yes", "on"}


@app.get("/api/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@app.get("/api/weights")
def get_default_weights() -> Response:
    return jsonify(dict(DEFAULT_WEIGHTS))


@app.get("/api/presets")
def get_presets() -> Response:
    return jsonify(WEIGHT_PRESETS)


def _process_request() -> Any:
    if "file" not in request.files:
        raise ValueError("Missing 'file' field in form-data.")
    upload = request.files["file"]
    raw = upload.read()
    if not raw:
        raise ValueError("Empty file.")
    df = _read_table(raw, filename=upload.filename or "")
    if df.empty:
        raise ValueError("Plik nie zawiera żadnych wierszy.")

    enrich = _get_form_bool("enrich")
    github_token = request.form.get("github_token") or None
    if enrich:
        df = enrich_dataframe(df, gh_token=github_token)

    weights = _parse_weights(request.form.get("weights"))
    return compute_ranking(df, weights)


@app.post("/api/rank")
def rank_endpoint() -> Response:
    try:
        result = _process_request()
    except ValueError as exc:
        return _json_error(str(exc))
    payload: Dict[str, Any] = {
        "weights": result.weights,
        "rows": json.loads(result.df.to_json(orient="records")),
        "columns": list(result.df.columns),
    }
    return jsonify(payload)


@app.post("/api/rank/csv")
def rank_csv_endpoint() -> Response:
    try:
        result = _process_request()
    except ValueError as exc:
        return _json_error(str(exc))
    buf = io.StringIO()
    result.df.to_csv(buf, index=False)
    resp = Response(buf.getvalue(), mimetype="text/csv")
    resp.headers["Content-Disposition"] = 'attachment; filename="models_ranked.csv"'
    return resp


@app.get("/")
def index() -> Response:
    return send_from_directory(FRONTEND_DIR, "index.html")


def _probe_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if a TCP connection to host:port can be opened."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# Candidate corporate proxies. First reachable one wins. Override with PROXY_CANDIDATES env.
DEFAULT_PROXY_CANDIDATES = (
    ("proxy-dmz.intel.com", 912),
    ("proxy-dmz.intel.com", 911),
    ("proxy-chain.intel.com", 912),
    ("proxy-chain.intel.com", 911),
)


def _parse_proxy_candidates(value: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece or ":" not in piece:
            continue
        host, port = piece.rsplit(":", 1)
        try:
            out.append((host.strip(), int(port.strip())))
        except ValueError:
            continue
    return out


def autodetect_proxy() -> Optional[str]:
    """Set HTTP(S)_PROXY env vars if a corporate proxy is reachable.

    Behavior:
      - If NO_PROXY_AUTODETECT is set (truthy), do nothing.
      - If HTTPS_PROXY/HTTP_PROXY is already set, keep it.
      - Else probe candidate proxies; first reachable -> set env vars.
    Returns the proxy URL set (or None).
    """
    log = logging.getLogger("ml-ranker.proxy")

    if os.environ.get("NO_PROXY_AUTODETECT", "").lower() in {"1", "true", "yes"}:
        log.info("Proxy autodetect disabled via NO_PROXY_AUTODETECT.")
        return os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    existing = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if existing:
        log.info("Using preset proxy from env: %s", existing)
        return existing

    raw = os.environ.get("PROXY_CANDIDATES")
    candidates = _parse_proxy_candidates(raw) if raw else list(DEFAULT_PROXY_CANDIDATES)

    for host, port in candidates:
        if _probe_tcp(host, port, timeout=2.0):
            url = f"http://{host}:{port}"
            os.environ["HTTPS_PROXY"] = url
            os.environ["HTTP_PROXY"] = url
            os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
            log.info("Proxy autodetected and enabled: %s", url)
            print(f"[proxy] reachable — enabled {url}")
            return url

    log.info("No corporate proxy reachable — running in DIRECT mode.")
    print("[proxy] none reachable — DIRECT mode")
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    autodetect_proxy()
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, debug=True)


if __name__ == "__main__":
    main()

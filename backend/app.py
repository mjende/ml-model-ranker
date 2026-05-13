"""Flask backend for ML Model Ranker.

Run:
    python -m backend.app
or:
    flask --app backend.app run --port 8000
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

from .enrichment import enrich_dataframe
from .ranker import DEFAULT_WEIGHTS, compute_ranking

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


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return pd.read_csv(io.BytesIO(file_bytes), sep=";")


def _get_form_bool(name: str) -> bool:
    val = request.form.get(name, "false").strip().lower()
    return val in {"1", "true", "yes", "on"}


@app.get("/api/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@app.get("/api/weights")
def get_default_weights() -> Response:
    return jsonify(dict(DEFAULT_WEIGHTS))


def _process_request() -> Any:
    if "file" not in request.files:
        raise ValueError("Missing 'file' field in form-data.")
    upload = request.files["file"]
    raw = upload.read()
    if not raw:
        raise ValueError("Empty file.")
    df = _read_csv(raw)
    if df.empty:
        raise ValueError("CSV has no rows.")

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


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, debug=bool(os.environ.get("DEBUG")))


if __name__ == "__main__":
    main()

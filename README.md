# ML Model Ranker

Aplikacja webowa do priorytetyzacji modeli ML/AI dla zespołów QA. Sortuje modele
według ważonej sumy znormalizowanych metryk: **popularność, architektura, rozmiar,
precyzja, szybkość inferencji, jakość dokumentacji**.

Bazuje na opisie i pseudokodzie z dokumentu projektowego „Ranking modeli ML –
projekt priorytetyzacji modeli dla QA”.

## Funkcje

- 📥 Import CSV z listą modeli (kolumna `Model` wymagana, reszta opcjonalna).
- 🌐 Opcjonalne wzbogacanie danych z **Hugging Face** i **GitHub** (downloads, likes, stars, parametry, architektura).
- ⚖️ Normalizacja: log + min-max, odwrócona min-max (dla rozmiaru/latencji), skala dyskretna (architektura).
- 🎚️ Konfigurowalne wagi kryteriów w UI.
- 🛡️ Obsługa braków danych – brakująca metryka ⇒ wagi pozostałych są renormalizowane (model nie jest karany).
- 📊 Tabela wyników z paskiem score, kolumnami `*_norm` i uzasadnieniem.
- ⬇️ Eksport posortowanego rankingu do CSV.

## Architektura

```
frontend/   – HTML + JS (waniliowy), serwowany przez Flask
backend/
  app.py        – Flask endpoints (/api/rank, /api/rank/csv, /api/weights)
  ranker.py     – normalizacja + ważone scoringi + uzasadnienia
  enrichment.py – integracja Hugging Face & GitHub API
```

Pipeline (jak w pseudokodzie):

1. **Ingest** – wczytanie CSV.
2. **Enrichment** – pobranie metryk z HF / GH (opcjonalnie).
3. **Normalization & Scoring** – log/min-max, ważona suma, sortowanie.
4. **Output** – JSON do UI lub CSV do pobrania.

## Uruchomienie

Wymagany Python 3.9+.

```bash
cd ml-model-ranker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m backend.app          # http://localhost:8000
```

Otwórz http://localhost:8000.

## Format CSV

### Wejście

| Kolumna         | Wymagane | Opis |
|-----------------|----------|------|
| `Model`         | tak      | Identyfikator (np. `meta-llama/Llama-2-7b`) |
| `architecture`  | nie      | `transformer`, `cnn`, `mamba`, `moe`, ... |
| `parameters`    | nie      | liczba parametrów |
| `accuracy`      | nie      | jakość/F1/BLEU (0–1 lub 0–100) |
| `downloads`     | nie      | pobrania (Hugging Face) |
| `stars`         | nie      | gwiazdki GitHub |
| `latency`       | nie      | czas inferencji [ms] |
| `documentation` | nie      | `full`/`partial`/`none` lub 0–1 |
| `github`        | nie      | repo w formie `owner/name` |

### Wyjście

`rank`, `model`, `score` (0–100), `justification`, oraz kolumny `*_norm`
ze znormalizowanymi wartościami komponentów.

## Wagi domyślne

| Kryterium        | Waga |
|------------------|------|
| Popularność      | 0.20 |
| Architektura     | 0.10 |
| Rozmiar          | 0.15 |
| Precyzja         | 0.30 |
| Szybkość         | 0.15 |
| Dokumentacja     | 0.10 |

## Licencja

MIT.

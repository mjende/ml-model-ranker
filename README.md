# ML Model Ranker

Aplikacja webowa do priorytetyzacji modeli ML/AI dla zespołów QA. Sortuje modele
według ważonej sumy znormalizowanych metryk: **popularność, architektura, rozmiar,
jakość, szybkość inferencji, dokumentacja, świeżość, licencja**.

Wynik jest dodatkowo korygowany przez **coverage** (frakcja kryteriów dostępnych
dla danego modelu) – modele z bardzo niepełnymi danymi nie wyprzedzają tych
ocenionych kompletnie.

Bazuje na opisie i pseudokodzie z dokumentu projektowego „Ranking modeli ML –
projekt priorytetyzacji modeli dla QA”.

## Funkcje

- 📥 Import **CSV** lub **XLSX** z listą modeli (kolumna `Model` / `Model name` wymagana, reszta opcjonalna).
- 🌐 Opcjonalne wzbogacanie danych z **Hugging Face** i **GitHub** (downloads, likes, stars, parametry, architektura, **licencja**, **data ostatniej aktualizacji**).
- ⚖️ 8 kryteriów: popularność, architektura (+ bonus za rzadkie architektury), rozmiar, jakość, szybkość, dokumentacja, **świeżość** (eksponencjalny zanik, półokres 365 dni), **licencja** (permisywność).
- 🎯 **Presety wag**: balanced, qa_focus, performance, research, production – wybór z dropdown lub edycja własna.
- 🛡️ Obsługa braków danych – brakująca metryka ⇒ wagi pozostałych są renormalizowane, a finalny score skalowany przez coverage (0.6–1.0).
- 🎚️ Wagi konfigurowane suwakami, zapamiętywane w localStorage.
- 📊 Tabela: sortowanie po kolumnach, filtr po nazwie/architekturze/licencji, filtr modality, toggle kolumn `*_norm`, kolorowy pasek score (red→green), badge coverage.
- 🔍 Modal ze szczegółami modelu po kliknięciu wiersza (z linkami do HF i GitHub).
- ⬇️ Eksport rankingu do **CSV** lub **JSON**.

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

### Proxy (sieci korporacyjne)

Serwer automatycznie wykrywa korporacyjny proxy przy starcie. Sprawdza po kolei:

1. Zmienne środowiskowe `HTTPS_PROXY` / `HTTP_PROXY` — jeśli ustawione, używane są bez zmian.
2. Listę kandydatów (domyślnie Intel: `proxy-dmz.intel.com:912/911`, `proxy-chain.intel.com:912/911`).
3. Jeśli żaden nie odpowiada → tryb **DIRECT**.

Ręczne nadpisanie:
```bash
PROXY_CANDIDATES="my-proxy.example.com:8080,backup-proxy:3128" python -m backend.app
NO_PROXY_AUTODETECT=1 python -m backend.app   # wyłącz autodetekcję całkowicie
```

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

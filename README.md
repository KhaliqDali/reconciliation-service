# SEA Cities Reconciliation Service

A [W3C Reconciliation Service API](https://www.w3.org/community/reconciliation/) for Southeast Asian cities, built so that [OpenRefine](https://openrefine.org/) can match messy city names in any dataset to authoritative [GeoNames](https://www.geonames.org/) records.

Rule based matching (exact, alternate names, trigram fuzzy) handles the easy cases in PostgreSQL. Sentence transformer embeddings and a Random Forest confidence scorer handle the ambiguous ones.

CM3070 Final Year Project, BSc Computer Science, University of London. Project Idea 2: Reconciliation service for a dataset.

## What it does

- Serves the W3C Reconciliation Service API (v0.2): manifest, reconciliation queries, suggest type, suggest property, properties and a flyout preview.
- Reference data: GeoNames `cities15000` filtered to the 11 Southeast Asian countries, cleaned in OpenRefine to 1,903 cities and 14,769 alternate names.
- Five layer matching pipeline: exact name, exact alternate name, trigram fuzzy on name, trigram fuzzy on alternate name, semantic similarity on `all-MiniLM-L6-v2` embeddings.
- Random Forest scorer on trigram similarity, semantic similarity and country match decides which fuzzy candidates are flagged as confident matches.
- Test framework of 39 checks against the specification (`test_api.py`).

## Repository layout

```
app/
  main.py                  FastAPI server and all W3C endpoints
  database.py              PostgreSQL connection from .env
  load_data.py             Loads data/SEA_cities_cleaned.csv into the cities table
  load_alternatenames.py   Splits alternate names into the alternatenames table
  generate_embeddings.py   Encodes all city names, writes data/city_embeddings.pkl
  train_model.py           Builds training pairs from the airports data, trains the Random Forest
  prepare_airports.py      Filters OpenFlights airports.dat to Southeast Asia
data/
  SEA_cities_cleaned.csv   Reference dataset (1,903 rows)
  SEA_airports.csv         Independent test set (363 rows)
  airports.dat.txt         Raw OpenFlights source
  city_embeddings.pkl      Precomputed embeddings
  confidence_model.pkl     Trained Random Forest
schema.sql                 Tables, pg_trgm extension and indexes
requirements.txt
test_api.py                W3C compliance test framework (39 checks)
test_semantic.py           Quick check of the semantic layer
```

## Setup

Tested on Windows 11 with Python 3.13, PostgreSQL 16 and OpenRefine 3.10.

1. Install PostgreSQL and create a database called `reconciliation_db`.
2. Run `schema.sql` against it (pgAdmin Query Tool or `psql -d reconciliation_db -f schema.sql`). This creates the two tables, enables `pg_trgm` and builds the trigram indexes.
3. Create `app/.env`:
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=reconciliation_db
   DB_USER=postgres
   DB_PASSWORD=your_password
   ```
4. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Load the data, from the project root:
   ```
   python app/load_data.py
   python app/load_alternatenames.py
   ```
6. The embeddings and the trained model are included in `data/`. To regenerate them:
   ```
   python app/generate_embeddings.py
   python app/train_model.py
   ```

## Running the server

From the project root:

```
uvicorn app.main:app
```

The server starts on `http://127.0.0.1:8000`. The manifest is at `http://127.0.0.1:8000/reconcile` and interactive docs at `http://127.0.0.1:8000/docs`.

Avoid `--reload` on machines with limited memory: it loads the sentence transformer twice.

## Connecting from OpenRefine

1. Open a project, click the dropdown on the column of city names, choose **Reconcile**, then **Start reconciling**.
2. Click **Add standard service** and enter `http://127.0.0.1:8000/reconcile`.
3. Select **Populated Place** as the type. If the project has a country column, tick it under "Also use relevant details from other columns" and set the property to `country`. The scorer uses it.
4. Click **Start reconciling**. Hover a matched cell to see the flyout with population, feature type, region, timezone and coordinates.

## Running the tests

With the server running:

```
python test_api.py
```

Expected output ends with `RESULTS: 39 passed / 39 total`.

## Results

Independent test set: 363 Southeast Asian airport cities from OpenFlights, reconciled inside OpenRefine with the country column supplied.

| Matching layers | Matched | Rate |
|---|---|---|
| Exact name only | 185 / 363 | 51% |
| + trigram fuzzy | 188 / 363 | 52% |
| + alternate names | 211 / 363 | 58% |
| + semantic layer + ML scorer | 220 / 363 | 61% |

Of the remaining rows, 121 received candidates that the scorer did not flag confidently enough for automatic acceptance, and 20 have no candidate because the place is below the 15,000 population cutoff of the source file.

The confidence model (3,610 candidate pairs, 206 positives) reaches precision 1.00 and recall 0.89 on the match class on a held out split. Feature importances: semantic similarity 0.428, trigram similarity 0.422, country match 0.151.

## Data sources

- GeoNames, `cities15000.txt`, Creative Commons Attribution 4.0. https://www.geonames.org/export/
- OpenFlights airport database, Open Database License. https://openflights.org/data

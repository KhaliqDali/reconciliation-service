from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.database import get_connection
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import traceback
import pickle
import numpy as np
import pandas as pd
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML models and embeddings on startup
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading city embeddings...")
with open('data/city_embeddings.pkl', 'rb') as f:
    embedding_data = pickle.load(f)

city_embeddings = np.array(embedding_data['embeddings'])
city_geonameids = embedding_data['geonameids']
city_names = embedding_data['names']
city_country_codes = embedding_data['country_codes']

print("Loading confidence scoring model...")
with open('data/confidence_model.pkl', 'rb') as f:
    confidence_model = pickle.load(f)

print(f"Loaded {len(city_geonameids)} city embeddings — AI models ready!")

SERVICE_MANIFEST = {
    "versions": ["0.2"],
    "name": "SEA Cities Reconciliation Service (AI-Powered)",
    "identifierSpace": "http://www.geonames.org/ontology#",
    "schemaSpace": "http://www.geonames.org/ontology#",
    "defaultTypes": [
        {"id": "PPL", "name": "Populated Place"}
    ],
    "suggest": {
        "property": {
            "service_url": "http://127.0.0.1:8000",
            "service_path": "/reconcile/suggest/property"
        },
        "type": {
            "service_url": "http://127.0.0.1:8000",
            "service_path": "/reconcile/suggest/type"
        }
    }
}

# Country name to ISO code mapping (mirrors the mapping used in training)
COUNTRY_MAP = {
    'philippines': 'PH', 'malaysia': 'MY', 'indonesia': 'ID',
    'thailand': 'TH', 'vietnam': 'VN', 'myanmar': 'MM',
    'singapore': 'SG', 'cambodia': 'KH', 'laos': 'LA',
    'brunei': 'BN', 'timor-leste': 'TL', 'east timor': 'TL'
}


def get_ml_confidence(query_string, city_name, semantic_score, country_match, cursor):
    """Score a candidate match using the trained Random Forest.

    Features are passed as a named DataFrame in the same column order
    used during training: semantic_score, trgm_score, country_match.
    country_match is now computed by the caller instead of hardcoded,
    since the model relies heavily on this feature.
    """
    cursor.execute("SELECT similarity(%s, %s)",
                  (query_string.lower(), city_name.lower()))
    trgm_score = float(cursor.fetchone()[0])
    features = pd.DataFrame(
        [[semantic_score, trgm_score, country_match]],
        columns=['semantic_score', 'trgm_score', 'country_match']
    )
    prob = confidence_model.predict_proba(features)[0]
    match_prob = prob[1] if len(prob) > 1 else prob[0]
    return float(match_prob), trgm_score


def _property_value_to_string(v):
    """Normalise an OpenRefine property value to a plain string.

    OpenRefine may send the value as a string, a dict ({"id": ..} or
    {"name": ..}), or a LIST of either — handle all of these.
    """
    if isinstance(v, list):
        v = v[0] if v else ""
    if isinstance(v, dict):
        v = v.get("id", v.get("name", ""))
    return str(v).strip()


def extract_query_country_code(query):
    """Read the country property OpenRefine sends alongside a query.

    Accepts either a 2-letter ISO code (e.g. "PH") or a full country
    name (e.g. "Philippines"), returning the ISO code or "" if absent.
    """
    for prop in query.get("properties", []):
        if prop.get("pid") in ("country_code", "country"):
            v = _property_value_to_string(prop.get("v", ""))
            if not v:
                continue
            if len(v) == 2:
                return v.upper()
            return COUNTRY_MAP.get(v.lower(), "")
    return ""


@app.get("/reconcile")
def reconcile_get():
    return JSONResponse(content=SERVICE_MANIFEST)

@app.post("/reconcile")
async def reconcile_post(request: Request):
    try:
        content_type = request.headers.get("content-type", "")

        if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form = await request.form()
            queries_raw = form.get("queries")
        else:
            body = await request.body()
            queries_raw = body.decode("utf-8")
            if queries_raw.startswith("queries="):
                queries_raw = queries_raw[8:]

        if not queries_raw:
            return JSONResponse(content=SERVICE_MANIFEST)

        queries = json.loads(queries_raw)
        results = {}

        conn = get_connection()
        cursor = conn.cursor()

        first_query_logged = False
        for key, query in queries.items():
            query_string = query.get("query", "")
            limit = query.get("limit", 5)
            query_country_code = extract_query_country_code(query)

            # TEMPORARY DEBUG — remove once country matching is verified
            if not first_query_logged:
                print(f"DEBUG raw query: {json.dumps(query)[:400]}")
                print(f"DEBUG extracted country code: '{query_country_code}'")
                first_query_logged = True

            # Layer 1 — exact match on name/asciiname
            cursor.execute("""
                SELECT c.geonameid, c.name, c.country_code,
                       c.feature_code, c.population, 1.0 AS score
                FROM cities c
                WHERE LOWER(c.name) = LOWER(%s)
                   OR LOWER(c.asciiname) = LOWER(%s)
                LIMIT %s
            """, (query_string, query_string, limit))
            exact_rows = cursor.fetchall()

            if exact_rows:
                rows_to_use = exact_rows
                use_ml_scoring = False
            else:
                # Layer 2 — exact match on alternate names
                cursor.execute("""
                    SELECT c.geonameid, c.name, c.country_code,
                           c.feature_code, c.population, 1.0 AS score
                    FROM cities c
                    JOIN alternatenames a ON a.geonameid = c.geonameid
                    WHERE LOWER(a.altname) = LOWER(%s)
                    LIMIT %s
                """, (query_string, limit))
                altname_exact = cursor.fetchall()

                if altname_exact:
                    rows_to_use = altname_exact
                    use_ml_scoring = False
                else:
                    # Layer 3 — fuzzy match on name/asciiname
                    cursor.execute("""
                        SELECT geonameid, name, country_code,
                               feature_code, population,
                               GREATEST(
                                   similarity(name, %s),
                                   similarity(asciiname, %s)
                               ) AS score
                        FROM cities
                        WHERE similarity(name, %s) > 0.3
                           OR similarity(asciiname, %s) > 0.3
                        ORDER BY score DESC
                        LIMIT %s
                    """, (query_string, query_string,
                          query_string, query_string, limit))
                    fuzzy_rows = cursor.fetchall()

                    if fuzzy_rows:
                        rows_to_use = fuzzy_rows
                        use_ml_scoring = True
                    else:
                        # Layer 4 — fuzzy match on alternate names
                        cursor.execute("""
                            SELECT DISTINCT c.geonameid, c.name,
                                   c.country_code, c.feature_code,
                                   c.population,
                                   similarity(a.altname, %s) AS score
                            FROM cities c
                            JOIN alternatenames a ON a.geonameid = c.geonameid
                            WHERE similarity(a.altname, %s) > 0.3
                            ORDER BY score DESC
                            LIMIT %s
                        """, (query_string, query_string, limit))
                        fuzzy_alt = cursor.fetchall()

                        if fuzzy_alt:
                            rows_to_use = fuzzy_alt
                            use_ml_scoring = True
                        else:
                            # Layer 5 — semantic similarity using ML embeddings
                            query_embedding = model.encode([query_string])
                            similarities = cosine_similarity(
                                query_embedding, city_embeddings
                            )[0]
                            top_indices = np.argsort(similarities)[::-1][:limit]

                            semantic_rows = []
                            for idx in top_indices:
                                if similarities[idx] > 0.5:
                                    gid = city_geonameids[idx]
                                    cursor.execute("""
                                        SELECT geonameid, name, country_code,
                                               feature_code, population
                                        FROM cities WHERE geonameid = %s
                                    """, (gid,))
                                    city = cursor.fetchone()
                                    if city:
                                        semantic_rows.append(
                                            city + (float(similarities[idx]),)
                                        )
                            rows_to_use = semantic_rows
                            use_ml_scoring = True

            candidates = []
            seen = set()
            for row in rows_to_use:
                geonameid, name, country_code, feature_code, population, score = row
                if geonameid not in seen:
                    seen.add(geonameid)

                    if use_ml_scoring:
                        query_emb = model.encode([query_string])
                        city_idx = city_geonameids.index(geonameid) if geonameid in city_geonameids else -1
                        if city_idx >= 0:
                            sem_score = float(cosine_similarity(
                                query_emb, [city_embeddings[city_idx]]
                            )[0][0])
                        else:
                            sem_score = 0.0

                        # Compute country_match for real (was hardcoded to 0,
                        # which made the model reject nearly every candidate)
                        country_match = 1 if (query_country_code and
                                              query_country_code == country_code) else 0

                        ml_conf, _ = get_ml_confidence(
                            query_string, name, sem_score, country_match, cursor
                        )
                        # TEMPORARY DEBUG — remove once country matching is verified
                        print(f"DEBUG ML: q='{query_string}' cand='{name}' "
                              f"country_match={country_match} conf={ml_conf:.3f}")
                        final_score = round(ml_conf * 100, 1)
                        # Threshold lowered from 0.9: with heavy class imbalance
                        # the forest is conservative, and verified true matches
                        # score in the 0.5-0.8 range
                        is_exact = ml_conf >= 0.5
                    else:
                        final_score = round(float(score) * 100, 1)
                        is_exact = float(score) >= 0.99

                    candidates.append({
                        "id": str(geonameid),
                        "name": f"{name} ({country_code})",
                        "score": final_score,
                        "match": is_exact,
                        "type": [{"id": feature_code, "name": feature_code}]
                    })

            results[key] = {"result": candidates}

        cursor.close()
        conn.close()

        return JSONResponse(content=results)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/reconcile/suggest/type")
def suggest_type(prefix: str = ""):
    types = [
        {"id": "PPL", "name": "Populated Place"},
        {"id": "PPLC", "name": "Capital City"},
        {"id": "PPLA", "name": "Provincial Capital"},
        {"id": "PPLA2", "name": "Secondary Capital"},
        {"id": "PPLA3", "name": "Third Level Capital"},
        {"id": "PPLX", "name": "Section of Populated Place"},
    ]
    filtered = [t for t in types if prefix.lower() in t["name"].lower()]
    return JSONResponse(content={"result": filtered})

@app.get("/reconcile/suggest/property")
def suggest_property(prefix: str = ""):
    properties = [
        {"id": "country_code", "name": "Country Code"},
        {"id": "population", "name": "Population"},
        {"id": "latitude", "name": "Latitude"},
        {"id": "longitude", "name": "Longitude"},
        {"id": "timezone", "name": "Timezone"},
        {"id": "feature_code", "name": "Feature Code"},
    ]
    filtered = [p for p in properties if prefix.lower() in p["name"].lower()]
    return JSONResponse(content={"result": filtered})

@app.get("/reconcile/properties")
def get_properties():
    return JSONResponse(content={
        "limit": 5,
        "type": "PPL",
        "properties": [
            {"id": "country_code", "name": "Country Code"},
            {"id": "population", "name": "Population"},
            {"id": "latitude", "name": "Latitude"},
            {"id": "longitude", "name": "Longitude"},
            {"id": "timezone", "name": "Timezone"},
            {"id": "feature_code", "name": "Feature Code"},
            {"id": "admin1_code", "name": "Admin Region"},
        ]
    })

@app.get("/reconcile/flyout/entity")
def flyout_entity(id: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, country_code, population,
               latitude, longitude, timezone,
               feature_code, admin1_code
        FROM cities
        WHERE geonameid = %s
    """, (id,))

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return JSONResponse(content={"html": "City not found"})

    name, country, population, lat, lon, timezone, feature_code, admin1 = row

    html = f"""
        <div style="font-family: Arial, sans-serif; padding: 10px;">
            <h3>{name} ({country})</h3>
            <table>
                <tr><td><b>Population</b></td><td>{population:,}</td></tr>
                <tr><td><b>Feature Type</b></td><td>{feature_code}</td></tr>
                <tr><td><b>Admin Region</b></td><td>{admin1}</td></tr>
                <tr><td><b>Timezone</b></td><td>{timezone}</td></tr>
                <tr><td><b>Coordinates</b></td><td>{lat}, {lon}</td></tr>
            </table>
        </div>
    """

    return JSONResponse(content={"html": html})
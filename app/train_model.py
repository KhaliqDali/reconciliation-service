import psycopg2
import pandas as pd
import numpy as np
import pickle
import re
import unicodedata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()

# Country name to ISO code mapping
COUNTRY_MAP = {
    'philippines': 'PH', 'malaysia': 'MY', 'indonesia': 'ID',
    'thailand': 'TH', 'vietnam': 'VN', 'myanmar': 'MM',
    'singapore': 'SG', 'cambodia': 'KH', 'laos': 'LA',
    'brunei': 'BN', 'timor-leste': 'TL', 'east timor': 'TL'
}


def strip_island_suffix(s):
    """Remove trailing geographic qualifiers found in the airports data.

    The airports dataset appends island qualifiers to some city names,
    e.g. 'Sampit-Borneo Island', 'Lhok Seumawe-Sumatra Island',
    'Batu Islands'. Stripping these for LABELLING (not for feature
    computation) creates training positives whose raw trgm score is
    low but whose semantic score is high — teaching the model to
    recognise this common real-world pattern instead of scoring it 0.
    """
    s = str(s)
    s = re.sub(r'[-\s]+[a-z]*\s*islands?$', '', s, flags=re.IGNORECASE)
    return s


def normalize_name(s):
    """Normalise a place name for comparison.

    Lowercases, strips diacritics (e.g. 'Nha Trang' vs 'Nhatrang',
    'Phú Quốc' vs 'Phu Quoc'), and removes all spaces, hyphens and
    punctuation. This lets genuinely matching name variants count as
    positive training examples even when their raw strings differ,
    so the model learns what real-world fuzzy matches look like
    instead of only recognising exact string equality.
    """
    s = str(s).lower().strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9]', '', s)
    return s


print("Connecting to database...")
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cursor = conn.cursor()

print("Loading city data...")
cursor.execute("SELECT geonameid, name, asciiname, country_code FROM cities")
cities = cursor.fetchall()
city_dict = {c[0]: c for c in cities}

print("Loading airports data...")
airports_df = pd.read_csv('data/SEA_airports.csv')
print(f"Airport countries: {airports_df['country'].unique()[:5]}")

print("Loading embeddings and model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
with open('data/city_embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
city_embeddings = np.array(emb_data['embeddings'])
city_geonameids = emb_data['geonameids']

print("Building training data...")
training_data = []
labels = []

for _, airport in airports_df.iterrows():
    query = str(airport['city'])
    if not query or query == 'nan':
        continue

    airport_country_name = str(airport['country']).lower().strip()
    airport_country_code = COUNTRY_MAP.get(airport_country_name, '')

    query_norm = normalize_name(query)
    query_clean_norm = normalize_name(strip_island_suffix(query))

    query_embedding = model.encode([query])
    sims = cosine_similarity(query_embedding, city_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:10]

    for idx in top_indices:
        gid = city_geonameids[idx]
        city = city_dict.get(gid)
        if not city:
            continue

        city_name = city[1]
        city_asciiname = city[2]
        city_country = city[3]

        semantic_score = float(sims[idx])

        cursor.execute("SELECT similarity(%s, %s)",
                      (query.lower(), city_name.lower()))
        trgm_score = float(cursor.fetchone()[0])

        country_match = 1 if airport_country_code == city_country else 0

        # Normalised name comparison against both name and asciiname.
        # This creates positive examples with trgm_score below 1.0
        # (e.g. 'Fak Fak' vs 'Fakfak', 'Nhatrang' vs 'Nha Trang'),
        # which is essential: the confidence model is only ever used
        # on fuzzy/semantic candidates, so it must see fuzzy positives
        # during training. The previous exact-equality labelling meant
        # every positive had trgm = 1.0, and the model learned to
        # reject all non-identical strings — scoring ~0.0 for every
        # candidate it was actually asked about in production.
        city_norm = normalize_name(city_name)
        city_ascii_norm = normalize_name(city_asciiname)
        name_match = (query_norm == city_norm or
                      query_norm == city_ascii_norm or
                      query_clean_norm == city_norm or
                      query_clean_norm == city_ascii_norm)
        is_match = 1 if (name_match and country_match == 1) else 0

        training_data.append([semantic_score, trgm_score, country_match])
        labels.append(is_match)

cursor.close()
conn.close()

df = pd.DataFrame(training_data, columns=[
    'semantic_score', 'trgm_score', 'country_match'
])
df['is_match'] = labels

print(f"\nTraining data: {len(df)} samples")
print(f"Positive matches: {df['is_match'].sum()}")
print(f"Negative matches: {len(df) - df['is_match'].sum()}")

# Show the trgm score distribution of positives — after the fix this
# should include values well below 1.0, proving fuzzy positives exist
positives = df[df['is_match'] == 1]
if len(positives) > 0:
    print(f"\nPositive examples trgm_score range: "
          f"{positives['trgm_score'].min():.3f} to "
          f"{positives['trgm_score'].max():.3f}")
    print(f"Positives with trgm_score < 0.99 (fuzzy positives): "
          f"{(positives['trgm_score'] < 0.99).sum()}")

if df['is_match'].sum() == 0:
    print("ERROR: Still no positive matches — check country mapping")
else:
    X = df[['semantic_score', 'trgm_score', 'country_match']]
    y = df['is_match']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining Random Forest classifier...")
    # class_weight='balanced' compensates for the severe imbalance
    # (~200 positives vs ~3,400 negatives) so the forest produces
    # usable probabilities for the minority class instead of
    # near-zero confidence for everything
    clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                 class_weight='balanced')
    clf.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("\nFeature importances:")
    for feat, imp in zip(X.columns, clf.feature_importances_):
        print(f"  {feat}: {imp:.3f}")

    print("\nSaving model...")
    with open('data/confidence_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Done! Model saved to data/confidence_model.pkl")
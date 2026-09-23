import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import pickle
import numpy as np

load_dotenv()

print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

cursor = conn.cursor()

print("Fetching city names...")
cursor.execute("SELECT geonameid, name, asciiname, country_code FROM cities")
rows = cursor.fetchall()

cursor.close()
conn.close()

print(f"Generating embeddings for {len(rows)} cities...")
geonameids = [row[0] for row in rows]
names = [f"{row[1]} {row[3]}" for row in rows]

embeddings = model.encode(names, show_progress_bar=True)

print("Saving embeddings to disk...")
data = {
    'geonameids': geonameids,
    'names': [row[1] for row in rows],
    'country_codes': [row[3] for row in rows],
    'embeddings': embeddings
}

with open('data/city_embeddings.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"Done! Saved {len(geonameids)} embeddings to data/city_embeddings.pkl")
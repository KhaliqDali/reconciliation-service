import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

cursor = conn.cursor()

def clean_int(val):
    try:
        return int(val) if val != '' else None
    except:
        return None

def clean_float(val):
    try:
        return float(val) if val != '' else None
    except:
        return None

print("Loading CSV...")
df = pd.read_csv("data/SEA_cities_cleaned.csv", 
    dtype=str,
    keep_default_na=False
)

print("Columns found:", list(df.columns))

print(f"Inserting {len(df)} rows...")
insert_query = """
    INSERT INTO cities (
        geonameid, name, asciiname, alternatenames,
        latitude, longitude, feature_class, feature_code,
        country_code, cc2, admin1_code, admin2_code,
        admin3_code, admin4_code, population, elevation,
        dem, timezone, modification_date
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (geonameid) DO NOTHING;
"""

rows = []
for _, row in df.iterrows():
    rows.append((
        clean_int(row['geonameid']),
        row['name'],
        row['asciiname'],
        row['alternatenames'],
        clean_float(row['latitude']),
        clean_float(row['longtitude']),
        row['feature_class'],
        row['feature_code'],
        row['country_code'],
        row['cc2'],
        row['admin1_code'],
        row['admin2_code'],
        row['admin3_code'],
        row['admin4_code'],
        clean_int(row['population']),
        clean_int(row['elevation']),
        clean_int(row['dem']),
        row['timezone'],
        row['modification_date'] if row['modification_date'] != '' else None
    ))

execute_batch(cursor, insert_query, rows, page_size=100)

conn.commit()
cursor.close()
conn.close()

print("Done! Data loaded successfully.")
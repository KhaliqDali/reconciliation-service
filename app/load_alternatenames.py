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

print("Fetching cities and alternate names...")
cursor.execute("SELECT geonameid, alternatenames FROM cities WHERE alternatenames != ''")
rows = cursor.fetchall()

print(f"Processing {len(rows)} cities...")
insert_query = """
    INSERT INTO alternatenames (geonameid, altname)
    VALUES (%s, %s)
    ON CONFLICT DO NOTHING;
"""

records = []
for geonameid, altnames in rows:
    if altnames:
        for altname in altnames.split(','):
            altname = altname.strip()
            if altname and len(altname) > 1:
                records.append((geonameid, altname))

print(f"Inserting {len(records)} alternate name records...")
execute_batch(cursor, insert_query, records, page_size=500)

conn.commit()
cursor.close()
conn.close()

print("Done! Alternate names loaded successfully.")
-- SEA Cities Reconciliation Service: database schema
-- Run against an empty database named reconciliation_db

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS cities (
    geonameid          INTEGER PRIMARY KEY,
    name               VARCHAR(200),
    asciiname          VARCHAR(200),
    alternatenames     TEXT,
    latitude           DOUBLE PRECISION,
    longitude          DOUBLE PRECISION,
    feature_class      CHAR(1),
    feature_code       VARCHAR(10),
    country_code       CHAR(2),
    cc2                VARCHAR(200),
    admin1_code        VARCHAR(20),
    admin2_code        VARCHAR(80),
    admin3_code        VARCHAR(20),
    admin4_code        VARCHAR(20),
    population         BIGINT,
    elevation          INTEGER,
    dem                INTEGER,
    timezone           VARCHAR(40),
    modification_date  DATE
);

CREATE TABLE IF NOT EXISTS alternatenames (
    id        SERIAL PRIMARY KEY,
    geonameid INTEGER REFERENCES cities(geonameid),
    altname   TEXT NOT NULL
);

-- Trigram indexes used by the fuzzy matching layers
CREATE INDEX IF NOT EXISTS idx_cities_name_trgm      ON cities USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_cities_asciiname_trgm ON cities USING gin (asciiname gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_altnames_altname_trgm ON alternatenames USING gin (altname gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_altnames_geonameid    ON alternatenames (geonameid);

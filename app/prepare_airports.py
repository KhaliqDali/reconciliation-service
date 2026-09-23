import pandas as pd

# OpenFlights airports.dat has no header row
# Column reference: https://openflights.org/data.html
columns = [
    'airport_id', 'name', 'city', 'country', 
    'iata', 'icao', 'latitude', 'longitude', 
    'altitude', 'timezone', 'dst', 'tz_database', 
    'type', 'source'
]

# Load the dataset
df = pd.read_csv(
    'data/airports.dat.txt', 
    header=None, 
    names=columns,
    encoding='utf-8'
)

# SEA countries to filter
sea_countries = [
    'Singapore', 'Malaysia', 'Indonesia', 'Thailand',
    'Vietnam', 'Philippines', 'Myanmar', 'Cambodia',
    'Laos', 'Brunei', 'Timor-Leste'
]

# Filter to SEA only
sea_airports = df[df['country'].isin(sea_countries)]

# Keep only useful columns
sea_airports = sea_airports[['airport_id', 'name', 'city', 'country', 'iata', 'latitude', 'longitude']]

# Save to CSV
sea_airports.to_csv('data/SEA_airports.csv', index=False)

print(f"Done! {len(sea_airports)} SEA airports saved to data/SEA_airports.csv")
print(sea_airports.head(10).to_string())
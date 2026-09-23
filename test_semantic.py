from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('data/city_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings = np.array(data['embeddings'])

test_queries = ['Saigon', 'Rangoon', 'Bangkok', 'Singapoer', 'KL']

for query_text in test_queries:
    query_embedding = model.encode([query_text])
    sims = cosine_similarity(query_embedding, embeddings)[0]
    top = np.argsort(sims)[::-1][:3]
    print(f"\nQuery: '{query_text}'")
    for i in top:
        print(f"  {data['names'][i]} ({data['country_codes'][i]}) - score: {sims[i]:.3f}")
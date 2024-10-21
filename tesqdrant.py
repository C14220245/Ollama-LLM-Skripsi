from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

qdrant_client = QdrantClient(
    "https://4f26ba36-6b99-401b-8176-443fb3ba13d0.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="M-u-W3daJ0GchaLQHWnKRgjmrYDtxdVNbssP4dTSFF5flr9eKsvy8A",
)

collection_name = "tes2"
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=100, distance=Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created.")
else:
    print(f"Collection '{collection_name}' already exists.")
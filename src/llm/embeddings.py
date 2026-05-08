"""Embeddings via gemini-embedding-001 (Google's current production embedding model).

The model defaults to 3072-dim but supports `output_dimensionality` to coerce
down. We pin to 768 here so the Neo4j vector index dimension stays stable —
if you change EMBED_DIM, you must drop and recreate `video_embeddings` and
`voice_embeddings` indexes (see data/seed/01_schema.cypher).

For production, swap to a higher-tier model or a dedicated embedding service
(Vertex AI text-embedding-005, OpenAI text-embedding-3-large, etc.) by
returning a different client from `get_embeddings_client`.
"""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..config import get_settings

EMBED_DIM = 768


def get_embeddings_client() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.GOOGLE_API_KEY,
        output_dimensionality=EMBED_DIM,
    )


async def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed a list of strings, batched. Returns EMBED_DIM-dim vectors in input order."""
    client = get_embeddings_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        result = await client.aembed_documents(chunk)
        out.extend(result)
    return out

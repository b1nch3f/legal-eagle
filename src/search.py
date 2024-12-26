import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery

service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
key = os.environ["AZURE_SEARCH_API_KEY"]
k_nearest_neighbors = 20


def get_embeddings(text: str):
    # There are a few ways to get embeddings. This is just one example.
    import openai

    open_ai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    open_ai_key = os.getenv("AZURE_OPENAI_API_KEY")

    client = openai.AzureOpenAI(
        azure_endpoint=open_ai_endpoint,
        api_key=open_ai_key,
        api_version="2023-05-15",
    )
    embedding = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return embedding.data[0].embedding


def single_vector_search(query):
    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(
        vector=get_embeddings(query),
        k_nearest_neighbors=k_nearest_neighbors,
        fields="chunkVector",
    )

    results = search_client.search(
        vector_queries=[vector_query],
        # select=["chunkId", "chunk"],
    )

    return results


if __name__ == "__main__":
    query = "List all the points presented by defense which led to acquittal"

    results = single_vector_search(query)

    for result in results:
        print(result)
    # single_vector_search_with_filter(query)
    # simple_hybrid_search(query)

import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
key = os.environ["AZURE_SEARCH_API_KEY"]
juddement_name = "CRLA-617-2010"


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


def get_documents():
    file_path = os.path.join("data", "extracts", f"{juddement_name}.txt")

    # Load document
    with open(file_path, "r", encoding="utf-8") as file:
        judgement = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=4000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([judgement])

    docs = []

    for index, item in tqdm(enumerate(texts)):
        chunk_id = f"{juddement_name}_{(index + 1):03}"
        parent_doc = juddement_name
        chunk = item.page_content
        cunk_vector = get_embeddings(item.page_content)

        item_dict = {
            "chunkId": chunk_id,
            "parentDoc": parent_doc,
            "chunk": chunk,
            "chunkVector": cunk_vector,
        }

        docs.append(item_dict)

    return docs


if __name__ == "__main__":
    credential = AzureKeyCredential(key)
    client = SearchClient(service_endpoint, index_name, credential)
    docs = get_documents()
    client.upload_documents(documents=docs)

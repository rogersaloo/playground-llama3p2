from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex

documents = SimpleDirectoryReader(
    "/home/raloo/pdev/playground-llama3p2/dataset",
    required_exts = [".pdf"],    
    recursive = True,
    ).load_data()


model_name = "Snowflake/snowflake-arctic-embed-m"
embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    trust_remote_code=True)

# Index and store in Qdrant
collection_name = "chat-with-c-pdf"

client = qdrant_client.QdrantClient(
    host="localhost", 
    port=6333
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes=documents,
    storage_context=storage_context
)

print(documents)
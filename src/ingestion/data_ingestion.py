import os
import bs4

os.environ["USER_AGENT"] = "TraceRAG/1.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("1. Loading the webpage...")
loader = WebBaseLoader(
    web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-title", "post-content")
        )
    ),
    bs_get_text_kwargs={"strip": True}
)
docs = loader.load()

if not docs or not docs[0].page_content.strip():
    raise ValueError("WebBaseLoader failed to extract any text content.")

print("2. Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = splitter.split_documents(docs)

if not chunks:
    raise ValueError("Text splitting step generated 0 chunks.")

print(f"   Generated {len(chunks)} text chunks.")

print("3. Generating embeddings and building Chroma DB...")
embedd = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedd,
    collection_name="agent_docs",
    persist_directory="./chroma_db",
)

# 4. Query Search Execution
query = "What are the core components of an LLM-powered agent?"
print(f"4. Running similarity search for query: '{query}'")
results = db.similarity_search(query, k=2)

print("\n--- Top Match ---")
if results:
    print(results[0].page_content)
else:
    print("No results found in vector database.")

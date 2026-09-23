from phoenix.otel import register
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from openinference.instrumentation.langchain import LangChainInstrumentor



load_dotenv()


# Setting up GROQ api key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# setting up Arize Phoenix tracer
tracer_provider = register()
LangChainInstrumentor().instrument()

# Loading Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Setting up the chroma db as retriever
vector_store = Chroma(
    collection_name="agent_docs",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Convert to a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY, 
    model="openai/gpt-oss-20b"  
)

# Creating Prompt
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know.

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Creating docs into strings
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create Chains
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run it
response = rag_chain.invoke("What are the core components of an LLM-powered agent?")
print(response)

tracer_provider.shutdown()



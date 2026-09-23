import nest_asyncio
nest_asyncio.apply()

import time
from src.generation.rag_pipeline import retriever, rag_chain, llm, embeddings
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision 
from ragas.run_config import RunConfig

# Setting up the question and ground truth
question = "What are the core components of an LLM-powered agent?"
ground_truth = "The core components are Planning, Memory, and Tool use."

# Fetching the retrieved docs and answer
retrieved_docs = retriever.invoke(question)
contexts = [doc.page_content for doc in retrieved_docs]
answer = rag_chain.invoke(question)

# Building the dataset
data = {
    "question" : [question],
    "answer" : [answer],
    "contexts" : [contexts],
    "ground_truth" : [ground_truth]
}

# Converting data dict to huggingface data 
eval_dataset = Dataset.from_dict(data)

# Evaluating
result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=llm,
    embeddings=embeddings,
    run_config=RunConfig(max_workers=1)
)

print(result)


time.sleep(3)

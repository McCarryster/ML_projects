from mistralai import Mistral
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

api_key = "api_key"
model = "ministral-3b-2410"

# Retrieve part
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={"device": "cuda"},  # or "cuda" for GPU
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = FAISS.load_local("/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_blender/data/data_types/processed_data/embeddings", embedding_model, allow_dangerous_deserialization=True)

# Query for search and generation
query = "Now how to anumate that cube? For example rotation animation?"

# This will embed the query internally and search
results = vectorstore.similarity_search(query, k=5)
excerpts = "\n\n".join([doc.page_content for doc in results])

# Simple prompt
prompt = f"""
[INST]
You are a helpful assistant specialized in answering questions about Blender. You will receive:
- A user question about Blender.
- A set of retrieved documentation excerpts relevant to the question.

Your task is to answer the user's question using only the information from the provided documentation excerpts. If the answer cannot be found in the excerpts, reply that the information is not available.

## User Question:
{query}

## Retrieved Documentation Excerpts:
{excerpts}

Please provide a clear and detailed step by step answer based strictly on the excerpts above. Always include Hotkeys of main actions if applicable.
[/INST]
"""

# Generation part
client = Mistral(api_key=api_key)
chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
)
print(chat_response.choices[0].message.content)
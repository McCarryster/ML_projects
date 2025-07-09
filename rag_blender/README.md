# Blender Docs RAG

A simple Retrieval-Augmented Generation (RAG) system over the official [Blender documentation](https://docs.blender.org/), using **BGE Large (English)** for embeddings, **LangChain** for retrieval, and **Mistral API** for generation.

---

## What This Is

This project lets you ask natural language questions about Blender and get context-aware answers grounded in the official documentation.

---

## Stack

- **Embeddings**: [`bge-large-en`](https://huggingface.co/BAAI/bge-large-en)
- **Retriever**: [LangChain](https://github.com/langchain-ai/langchain)
- **Vector store** `FAISS`
- **LLM (generation)**: [Mistral 3B API](https://mistral.ai/)
- **Data**: Cleaned HTML from [docs.blender.org](https://docs.blender.org/)

---

## Project Structure

```
blender-docs-rag/
├── data/data_processing/   # Cleaning and vectorization
├── scripts/rag.py          # Main RAG script (retrieval + generation)
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/McCarryster/ML_projects/tree/master/rag_blender
cd rag_blender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

- `langchain`
- `sentence-transformers`
- `faiss-cpu`
- `requests` (for Mistral API)

### 3. Prepare data

Ensure cleaned Blender HTML docs are in the `data/data_types/raw_data` directory. You can clean and chunk them however you like

### 4. Generate embeddings

Change paths in `data/data_processing/` scripts to your preferences

```bash
python 2_vectorize_data.py
```

This will process the docs and store embeddings in a FAISS index (or other vector store).

### 5. Ask questions

Write your query in `scripts/rag.py`

```bash
python rag.py
```

---

## Example

```text
Q: How do I bake a normal map in Blender?

→ [retrieves relevant documentation sections]
→ [sends context + question to Mistral API]
→ A: "To bake a normal map in Blender, go to the Render tab..."
```

---

## License

MIT License. Attribution to Blender, Mistral, and BAAI for the models and documentation.

---

## Acknowledgments

- [Blender Docs](https://docs.blender.org/)
- [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en)
- [Mistral](https://mistral.ai/)
- [LangChain](https://github.com/langchain-ai/langchain)
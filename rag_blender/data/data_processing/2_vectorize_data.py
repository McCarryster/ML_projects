from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from mistralai import Mistral
from langchain_community.vectorstores.faiss import FAISS

loader = DirectoryLoader(
    path="/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_blender/data/data_types/processed_data/cleaned_html",
    glob="**/*.html",
    loader_cls=BSHTMLLoader,
    loader_kwargs={"open_encoding": "utf-8"},
    show_progress=True
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256)
splits = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={"device": "cuda"},  # or "cuda" for GPU
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = FAISS.from_documents(splits, embedding_model)

# Save FAISS index locally to a directory
vectorstore.save_local("/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_blender/data/data_types/processed_data/embeddings")
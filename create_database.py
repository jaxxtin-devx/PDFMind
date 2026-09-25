from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DB_PATH = "Chroma-DB"
COLLECTION_NAME = "dna_book"
PDF_PATH = "document_loader/DNA.pdf"

Path(DB_PATH).mkdir(exist_ok=True)

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)
chunks = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=DB_PATH,
    collection_name=COLLECTION_NAME,
)

print(f"Database created successfully with {len(chunks)} chunks at {DB_PATH} in collection '{COLLECTION_NAME}'")
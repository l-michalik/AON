from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import openai
import shutil
import os

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

CHROMA_PATH = "chroma"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data/books")

def main():
    """Main entry point: generates the Chroma vector store."""
    generate_data_store()

def generate_data_store():
    """Loads, splits, and stores documents in Chroma vector store."""
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents() -> list[Document]:
    """Loads .txt documents from the specified data directory."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"DATA_PATH does not exist: {DATA_PATH}")
    
    print(f"Loading documents from {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    return loader.load()

def split_text(documents: list[Document]) -> list[Document]:
    """Splits documents into smaller chunks for vector storage."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Show example chunk
    example = chunks[10]
    print("\nSample chunk content:")
    print(example.page_content)
    print("Metadata:", example.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    """Saves text chunks to a Chroma vector database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()

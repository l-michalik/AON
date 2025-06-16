from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import argparse
import openai
import os

# Load environment variables
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Constants
CHROMA_PATH = "chroma"
RELEVANCE_THRESHOLD = 0.7  # Minimum score to accept result

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    """
    Loads a persisted Chroma vector store, searches for relevant context
    based on a user query, and uses an LLM to answer using only that context.
    """
    # Parse command-line input
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The question to ask.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load Chroma vector store with OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search the database for top 3 most relevant chunks
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if not results or results[0][1] < RELEVANCE_THRESHOLD:
        print("Unable to find matching results with high enough relevance.")
        return

    # Combine the content of the most relevant chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Create a prompt using the retrieved context
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text
    )
    print("\nGenerated prompt:\n")
    print(prompt)

    # Generate a response using ChatOpenAI
    llm = ChatOpenAI()
    response = llm.invoke(prompt)

    # Collect sources from metadata (if any)
    sources = [doc.metadata.get("source", "Unknown") for doc, _ in results]

    # Print the result
    print("\nFinal Response:\n")
    print(f"{response}\n\nSources: {sources}")

if __name__ == "__main__":
    main()

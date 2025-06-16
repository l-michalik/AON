# Import required libraries
from langchain.evaluation import load_evaluator
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

def main():
    """
    Generates an embedding vector for a word using OpenAI,
    and compares embeddings of two words using a built-in evaluator.
    """
    
    # Create embedding function
    embeddings = OpenAIEmbeddings()

    # Get embedding for the word "apple"
    vector = embeddings.embed_query("apple")
    print("Embedding for 'apple':")
    print(vector)
    print("Vector length:", len(vector))

    # Compare embeddings for "apple" and "iphone"
    evaluator = load_evaluator("pairwise_embedding_distance")
    result = evaluator.evaluate_string_pairs(prediction="apple", prediction_b="iphone")
    print("\nSimilarity between 'apple' and 'iphone':")
    print(result)

# Run the main function
if __name__ == "__main__":
    main()

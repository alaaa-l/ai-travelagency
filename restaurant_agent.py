from embeddings import embed_texts
from similarity import retrieve_relevant_chunks
from call_llm import generate_answer
import os
from dotenv import load_dotenv

load_dotenv()

def get_restaurants(country: str, rag_collection, top_k: int = 5) -> str:
    """
    Returns a list of recommended restaurants for a country.
    """

    # 1. Embed query
    query = f"Best restaurants to visit in {country}"
    query_vector = embed_texts([query])

    # 2. Retrieve relevant chunks
    result = retrieve_relevant_chunks(
        query_vector,
        rag_collection,
        top_k=top_k
    )

    context = "\n".join(result["documents"][0])

    system_prompt = """
    You are a travel assistant.
    Use ONLY the provided context.
    
    Return a short bullet list of restaurants.
    Do NOT add explanations.
    If none are found, say "No restaurants found".
    """

    final_prompt = f"""
    Context:
    {context}

    Country:
    {country}
    """

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not found")

    return generate_answer(
        system_prompt + "\n" + final_prompt,
        api_key
    ).strip()

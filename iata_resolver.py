from embeddings import embed_texts
from similarity import retrieve_relevant_chunks
from prompt import prepare_prompt
from call_llm import generate_answer
import os
from dotenv import load_dotenv

load_dotenv()

def get_main_iata_code(location_name: str, rag_collection) -> str:
    """
    Resolves a city or country name to the IATA code of the main airport only.
    Returns just the code as a string (e.g., "CDG" or "BEY").
    """

    # 1. Embed the query
    query_vector = embed_texts([location_name])

    # 2. Retrieve relevant chunks from RAG
    result = retrieve_relevant_chunks(
        query_vector,
        rag_collection,
        top_k=5
    )

    # 3. Build a strict prompt to avoid extra info
    system_prompt = """
    You are a travel data assistant.
    Use ONLY the provided context to answer.
    
    Return ONLY the IATA code of the MAIN airport in the city.
    Do NOT add any extra text, explanation, or JSON.
    If no code is found, return "NOT FOUND".
    """

    context = "\n".join(result["documents"][0])

    final_prompt = f"""
    Context:
    {context}

    Location query:
    {location_name}
    """

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not found in .env")

    # 4. Call Deepseek
    code = generate_answer(
        system_prompt + "\n" + final_prompt,
        api_key
    ).strip()

    return code

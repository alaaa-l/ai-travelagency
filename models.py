import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
def get_deepseek():
    """
    returns a model to invoke DeepSeek
    """
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    url = os.getenv("DEEPSEEK_BASE_URL")

    llm_model = ChatOpenAI(
        model="deepseek-chat",
        max_tokens=1000,
        timeout=30,
        api_key=deepseek_key,
        base_url=url
    )
    return llm_model

def get_gemini():
    """
    returns a model to invoke Google Gimini.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=gemini_key,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    return llm_model
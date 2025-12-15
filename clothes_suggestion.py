import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages  import SystemMessage, HumanMessage

load_dotenv()
llm = init_chat_model(model="deepseek-chat")

def clothes_suggestions(weather:str):
    system_prompt = """
    You are a clothes suggester AI agent.
    Your task is to recommend clothes based on the weather.
    Return ONLY the recommended clothes in short form.
    
    
    The answer must be short and business-like:
    Return ONLY the recommended clothes.
    """

    user_prompt = f"Weather summary: {weather}\nSuggest suitable clothes to wear."

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return response.content
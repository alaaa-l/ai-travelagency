import os
from models import get_deepseek
import requests 
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    response = requests.get(f'https://wttr.in/{location}?format=j1')
    return response.json()

def weather_agent(country: str):
    agent_tools = [get_weather]

    system_prompt = """
    You are a weather assistant.
    When given a country, you MUST call the get_weather tool.
    After receiving the weather data, summarize it clearly.
    """

    agent = create_agent(
        model=get_deepseek(),
        tools=agent_tools,
        system_prompt=system_prompt
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"What is the current weather in {country}?"
                }
            ]
        }
    )

    return result["messages"][-1].content.strip()
    
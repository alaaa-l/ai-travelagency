import os
from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages  import SystemMessage, HumanMessage

load_dotenv()
llm = init_chat_model(model="deepseek-chat")

def recommend_country_agent(budget, interests, previous_destinations, duration):
    system_prompt = """
    You are a travel-planning AI agent. 
    Your task is to analyze user constraints and recommend ONE country 
    that best matches the user's:
    - budget
    - interests
    - prior travel history
    - trip duration
    
    The answer must be short and business-like:
    Return ONLY the chosen country without any more information.
    """

    user_prompt = f"""
    Budget: {budget}
    Interests: {interests}
    Previous destinations: {previous_destinations}
    Trip duration: {duration} days

    Based on this information, which ONE country is the best match?
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return response.content


# Example call
print(
    recommend_country_agent(
        budget="1500 USD",
        interests="beaches, nightlife",
        previous_destinations="France, Turkey",
        duration=7
    )
)
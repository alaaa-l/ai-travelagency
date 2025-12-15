from dotenv import load_dotenv

load_dotenv()
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from recommended_country_agent import recommend_country_agent
from iata_resolver import get_main_iata_code
from weather_agent import weather_agent
from clothes_suggestion import clothes_suggestions

from travel_cost_agent import get_travel_cost
from rag import my_rag_collection



def get_travel_workflow_app():
    # ---------------- Step 1: State Definition ----------------
    class TravelState(TypedDict):
        user_info: dict
        country: str
        iata_code: str
        weather_summary: str
        clothes: str
        travel_cost: int
        itinerary: str
        messages: list

    # ---------------- Step 2: Nodes Definition ----------------
    def user_input_node(state: TravelState):
        return {"messages": [HumanMessage(content="User info received")], "next_step": "Country_Recommendation"}

    def country_recommendation_node(state: TravelState):
        country = recommend_country_agent(
            budget=state['user_info']['budget'],
            interests=state['user_info']['interests'],
            previous_destinations=state['user_info']['previous_destinations'],
            duration=state['user_info']['duration']
        )
        return {
            "country": country,
            "next_step": "IATA_Resolver",
           "messages": [AIMessage(content=f"Recommended country: {country}")]
        } 
         


    def iata_node(state: TravelState):
        iata = get_main_iata_code(state['country'], rag_collection=my_rag_collection)
        return {"iata_code": iata, "next_step": "Weather_Node",
                "messages": [AIMessage(content=f"IATA: {iata}")]}

    def weather_node(state: TravelState):
        weather = weather_agent(state['country'])
        return {"weather_summary": weather, "next_step": "Clothes_Node",
                "messages": [AIMessage(content=f"Weather: {weather}")]}

    def clothes_node(state: TravelState):
        clothes = clothes_suggestions(state['weather_summary'])
        return {"clothes": clothes, "next_step": "Travel_Cost_Node",
                "messages": [AIMessage(content=f"Clothes: {clothes}")]}

    def travel_cost_node(state: TravelState):
        cost = get_travel_cost(state['user_info']['origin'], state['iata_code'], state['user_info']['date'])
        return {"travel_cost": cost, "next_step": "Budget_Check_Node",
                "messages": [AIMessage(content=f"Travel cost: {cost}")]}

    def budget_check_node(state: TravelState):
        budget = float(state['user_info']['budget'])
        if state['travel_cost'] > budget:
            return {"next_step": "Itinerary_Adjust_Node",
                    "messages":[AIMessage(content="Budget exceeded, need to adjust itinerary")]}
        return {"next_step": "Itinerary_Node",
                "messages":[AIMessage(content="Budget OK")]}

    def itinerary_node(state: TravelState):
        itinerary = f"Planned {state['user_info']['duration']}-day trip in {state['country']} with suggested activities."
        return {"itinerary": itinerary, "next_step": "Supervisor_Node",
                "messages":[AIMessage(content=itinerary)]}

    def itinerary_adjust_node(state: TravelState):
        itinerary = f"Adjusted {max(state['user_info']['duration']-1,1)}-day trip in {state['country']} to fit budget."
        return {"itinerary": itinerary, "next_step": "Final_Summary_Node",
                "messages":[AIMessage(content=itinerary)]}

    def supervisor_node(state: TravelState):
        budget = float(state['user_info']['budget'])
        if state['travel_cost'] > budget:
            return {"next_step": "Itinerary_Adjust_Node"}
        return {"next_step": "Final_Summary_Node"}

    def final_summary_node(state: TravelState):
        summary = f"""
Country: {state.get('country', '')}
IATA: {state.get('iata_code', '')}
Weather: {state.get('weather_summary', '')}
Clothes: {state.get('clothes', '')}
Travel Cost: {state.get('travel_cost', '')}
Itinerary: {state.get('itinerary', '')}
"""
        return {"messages":[AIMessage(content=summary)], "next_step": END}

    # ---------------- Step 3: Build Graph ----------------
    graph = StateGraph(TravelState)

    nodes = [
        ("User_Input", user_input_node),
        ("Country_Recommendation", country_recommendation_node),
        ("IATA_Resolver", iata_node),
        ("Weather_Node", weather_node),
        ("Clothes_Node", clothes_node),
        ("Travel_Cost_Node", travel_cost_node),
        ("Budget_Check_Node", budget_check_node),
        ("Itinerary_Node", itinerary_node),
        ("Itinerary_Adjust_Node", itinerary_adjust_node),
        ("Supervisor_Node", supervisor_node),
        ("Final_Summary_Node", final_summary_node)
    ]

    for name, fn in nodes:
        graph.add_node(name, fn)

    # Entry point
    graph.set_entry_point("User_Input")

    # Conditional routing
    graph.add_edge("User_Input", "Country_Recommendation")
    graph.add_edge("Country_Recommendation", "IATA_Resolver")
    graph.add_edge("IATA_Resolver", "Weather_Node")
    graph.add_edge("Weather_Node", "Clothes_Node")
    graph.add_edge("Clothes_Node", "Travel_Cost_Node")
    graph.add_edge("Travel_Cost_Node", "Budget_Check_Node")
    graph.add_edge("Budget_Check_Node", "Itinerary_Node")
    graph.add_edge("Budget_Check_Node", "Itinerary_Adjust_Node")
    graph.add_edge("Itinerary_Node", "Supervisor_Node")
    graph.add_edge("Itinerary_Adjust_Node", "Final_Summary_Node")
    graph.add_edge("Supervisor_Node", "Itinerary_Adjust_Node")
    graph.add_edge("Supervisor_Node", "Final_Summary_Node")

    # Compile workflow
    return graph.compile()

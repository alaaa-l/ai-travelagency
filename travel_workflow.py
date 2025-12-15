from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage


from recommended_country_agent import recommend_country_agent
from iata_resolver import get_main_iata_code
from weather_agent import weather_agent
from clothes_suggestion import clothes_suggestions
from travel_cost_agent import get_travel_cost

from rag import get_rag_collection

# Initialize RAG collection once
rag_collection = get_rag_collection()

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
        adjusted_itinerary: str  # separate key to avoid LangGraph conflict
        messages: list
        rag_collection: any
        budget_adjustments: int  # Track how many times we adjusted itinerary

    # ---------------- Step 2: Nodes Definition ----------------
    def user_input_node(state: TravelState):
        return {
            "messages": state.get("messages", []) + [HumanMessage(content="User info received")],
            "next_step": "Country_Recommendation"
        }

    def country_recommendation_node(state: TravelState):
        country = recommend_country_agent(
            budget=state['user_info']['budget'],
            interests=state['user_info']['interests'],
            previous_destinations=state['user_info']['previous_destinations'],
            duration=state['user_info']['duration']
        )
        return {
            "country": country,
            "messages": state.get("messages", []) + [AIMessage(content=f"Recommended country: {country}")],
            "next_step": "IATA_Resolver"
        }

    def iata_node(state: TravelState):
        iata = get_main_iata_code(state['country'], rag_collection=state['rag_collection'])
        return {
            "iata_code": iata,
            "messages": state.get("messages", []) + [AIMessage(content=f"IATA: {iata}")],
            "next_step": "Weather_Node"
        }

    def weather_node(state: TravelState):
        weather = weather_agent(state['country'])
        return {
            "weather_summary": weather,
            "messages": state.get("messages", []) + [AIMessage(content=f"Weather: {weather}")],
            "next_step": "Clothes_Node"
        }

    def clothes_node(state: TravelState):
        clothes = clothes_suggestions(state['weather_summary'])
        return {
            "clothes": clothes,
            "messages": state.get("messages", []) + [AIMessage(content=f"Clothes: {clothes}")],
            "next_step": "Travel_Cost_Node"
        }

    def travel_cost_node(state: TravelState):
        cost = get_travel_cost(state['user_info']['origin'], state['iata_code'], state['user_info']['date'])
        return {
            "travel_cost": cost,
            "messages": state.get("messages", []) + [AIMessage(content=f"Travel cost: {cost}")],
            "next_step": "Budget_Check_Node"
        }

    def budget_check_node(state: TravelState):
        # Limit max 1 adjustment to prevent infinite loop
        adjustments = state.get("budget_adjustments", 0)
        budget = float(state['user_info']['budget'])
        if state['travel_cost'] > budget and adjustments < 1:
            return {
                "next_step": "Itinerary_Adjust_Node",
                "budget_adjustments": adjustments + 1,
                "messages": state.get("messages", []) + [AIMessage(content="Budget exceeded, adjusting itinerary")]
            }
        return {
            "next_step": "Itinerary_Node",
            "messages": state.get("messages", []) + [AIMessage(content="Budget OK")]
        }

    def itinerary_node(state: TravelState):
        itinerary = f"Planned {state['user_info']['duration']}-day trip in {state['country']} with suggested activities."
        return {
            "itinerary": itinerary,
            "messages": state.get("messages", []) + [AIMessage(content=itinerary)],
            "next_step": "Supervisor_Node"
        }

    def itinerary_adjust_node(state: TravelState):
        adjusted_itinerary = f"Adjusted {max(state['user_info']['duration']-1,1)}-day trip in {state['country']} to fit budget."
        return {
            "adjusted_itinerary": adjusted_itinerary,
            "messages": state.get("messages", []) + [AIMessage(content=adjusted_itinerary)],
            "next_step": "Final_Summary_Node"
        }

    def supervisor_node(state: TravelState):
        budget = float(state['user_info']['budget'])
        if state['travel_cost'] > budget:
            return {"next_step": "Itinerary_Adjust_Node"}
        return {"next_step": "Final_Summary_Node"}

    def final_summary_node(state: TravelState):
        # Use adjusted itinerary if available
        itinerary = state.get("adjusted_itinerary") or state.get("itinerary", "")
        summary = f"""
Country: {state.get('country', '')}
IATA: {state.get('iata_code', '')}
Weather: {state.get('weather_summary', '')}
Clothes: {state.get('clothes', '')}
Travel Cost: {state.get('travel_cost', '')}
Itinerary: {itinerary}
"""
        return {
            "messages": state.get("messages", []) + [AIMessage(content=summary)],
            "next_step": END
        }

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

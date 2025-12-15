import operator
from typing import Annotated, List, TypedDict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

#Function we implement
from recommended_country_agent import recommend_country_agent
from iata_resolver import get_main_iata_code
from weather_agent import weather_agent
from clothes_suggestion import clothes_suggestions
from travel_cost_agent import get_travel_cost
from hotel_agent import get_hotels  
from restaurant_agent import get_restaurants 
from rag import get_rag_collection
from models import get_deepseek 

# --- Global Initialization ---
RAG_COLLECTION = get_rag_collection()
LLM = get_deepseek() # Initialize LLM

# ---------------- Step 1: State Definition ----------------
class TravelState(TypedDict):
    """
    Represents the full state of the travel planning workflow.
    """
    next_step: str 
    
    user_info: dict
    country: str
    iata_code: str # Destination IATA
    origin_iata_code: str #Origin IATA 
    weather_summary: str
    clothes: str
    travel_cost: float 
    
    hotel_options: str 
    restaurant_options: str
    
    itinerary: str
    budget_adjustments: int 
    
    messages: Annotated[List[BaseMessage], operator.add] 
    rag_collection: Any

# ---------------- Step 2: Nodes Definition ----------------

# Node 1: Initial setup 
def user_input_node(state: TravelState):
    """Initializes the RAG collection, LLM, and sets the starting route."""
    print("--- PREP: Initializing State ---")
    return {
        "messages": state.get("messages", []) + [AIMessage(content="User info received and flow initiated.")],
        "rag_collection": RAG_COLLECTION,
        "budget_adjustments": state.get("budget_adjustments", 0),
        "next_step": "Country_Recommendation"
    }

# Node 2: WORKER: Recommends Country 
def country_recommendation_node(state: TravelState):
    print("--- WORKER: Recommending Country ---")
    country = recommend_country_agent(
        llm=LLM, 
        budget=state['user_info']['budget'],
        interests=state['user_info']['interests'],
        previous_destinations=state['user_info']['previous_destinations'],
        duration=state['user_info']['duration'],
        adjustment_attempt=state['budget_adjustments']
    )
    return {
        "country": country,
        "messages": [AIMessage(content=f"Recommended country: {country}")],
        "next_step": "IATA_Resolver"
    }

# Node 3: WORKER: Resolves Destination IATA Code (Routes to Origin Resolver)
def iata_node(state: TravelState):
    print("--- WORKER: Resolving Destination IATA Code ---")
    iata = get_main_iata_code(state['country'], rag_collection=state['rag_collection'])
    return {
        "iata_code": iata,
        "messages": [AIMessage(content=f"Destination IATA code resolved: {iata}")],
        "next_step": "Origin_IATA_Resolver" 
    }

# Node 4 : WORKER: Resolves Origin IATA Code 
def origin_iata_node(state: TravelState):
    print("--- WORKER: Resolving Origin IATA Code ---")
    
    origin_country_name = state['user_info']['origin'] 
    
    origin_iata = get_main_iata_code(origin_country_name, rag_collection=state['rag_collection'])
    
    return {
        "origin_iata_code": origin_iata, 
        "messages": [AIMessage(content=f"Origin IATA: {origin_iata}")],
        "next_step": "Travel_Cost_Node"
    }


# Node 5: WORKER: Calculates Travel Cost 
def travel_cost_node(state: TravelState):
    print("--- WORKER: Calculating Travel Cost (Primary Constraint) ---")
    
    # Access both IATA codes from the state
    origin_iata = state['origin_iata_code'] 
    destination_iata = state['iata_code'] 
    
    cost = get_travel_cost(
        origin=origin_iata, 
        destination_iata=destination_iata,
        date=state['user_info']['date']
    )
    
    return {
        "travel_cost": float(cost),
        "messages": [AIMessage(content=f"Travel cost: ${cost}")],
        "next_step": "Budget_Check_Node" 
    }
    
# Node 6: SUPERVISOR/ROUTER: Checks Budget 
def budget_check_node(state: TravelState):
    print("--- SUPERVISOR: Checking Budget Constraint ---")
    
    adjustments = state.get("budget_adjustments", 0)
    budget = float(state['user_info']['budget'])
    
    if state['travel_cost'] > budget and adjustments < 2:
        print(f"-> Decision: Budget exceeded (${state['travel_cost']} > ${budget}). Routing back to Country Rec.")
        return "Country_Recommendation_Retry"
        
    print("-> Decision: Budget OK or max adjustments hit. Routing to Weather/Plan.")
    return "Weather_Node"

# Nodes 7-12 (Weather, Clothes, Hotel, Restaurant, Itinerary, Summary) remain as you defined them.

def weather_node(state: TravelState):
    print("--- WORKER: Fetching Weather ---")
    weather = weather_agent(state['country'])
    return {
        "weather_summary": weather, "messages": [AIMessage(content=f"Weather: {weather}")], "next_step": "Clothes_Node"
    }

def clothes_node(state: TravelState):
    print("--- WORKER: Suggesting Clothes ---")
    clothes = clothes_suggestions(state['weather_summary'])
    return {
        "clothes": clothes, "messages": [AIMessage(content=f"Clothes suggestion: {clothes}")], "next_step": "Hotel_Node"
    }

def hotel_node(state: TravelState):
    print("--- WORKER: Suggesting Hotels ---")
    hotels = get_hotels(
        country=state['country'], rag_collection=state['rag_collection'], top_k=5
    )
    return {
        "hotel_options": str(hotels), "messages": [AIMessage(content=f"Hotels found.")], "next_step": "Restaurant_Node"
    }

def restaurant_node(state: TravelState):
    print("--- WORKER: Suggesting Restaurants ---")
    restaurants = get_restaurants(
        country=state['country'], rag_collection=state['rag_collection'], top_k=5
    )
    return {
        "restaurant_options": str(restaurants), "messages": [AIMessage(content=f"Restaurants found.")], "next_step": "Itinerary_Node"
    }

def itinerary_node(state: TravelState):
    print("--- WORKER: Final Itinerary Assembly ---")
    itinerary_text = f"""
    A {state['user_info']['duration']}-day trip to {state['country']} is planned. 
    Weather summary: {state['weather_summary']}
    Hotel suggestions: {state['hotel_options']}
    Restaurant suggestions: {state['restaurant_options']}
    Final cost estimate: ${state['travel_cost']}
    """
    return {
        "itinerary": itinerary_text, "messages": [AIMessage(content="Final itinerary assembled.")], "next_step": "Final_Summary_Node" 
    }

def final_summary_node(state: TravelState):
    print("--- WORKER: Finalizing Plan Summary ---")
    summary = f"""
    *** FINAL TRAVEL PLAN ***
    -------------------------
    Destination: {state.get('country', 'N/A')} 
    Total Estimated Cost: ${state.get('travel_cost', 'N/A')}
    --- Detailed Plan ---
    {state.get('itinerary', 'Plan created.')}
    -------------------------
    """
    return {
        "messages": [AIMessage(content=summary)], "next_step": "FINISH" 
    }


def get_travel_workflow_app():
    # ---------------- Step 3: Build Graph ----------------
    graph = StateGraph(TravelState)

    # ---------------- Add nodes ----------------------
    nodes = [
        ("User_Input", user_input_node),
        ("Country_Recommendation", country_recommendation_node),
        ("IATA_Resolver", iata_node),
        ("Origin_IATA_Resolver", origin_iata_node), 
        ("Travel_Cost_Node", travel_cost_node), 
        ("Budget_Check_Node", budget_check_node),
        ("Weather_Node", weather_node),
        ("Clothes_Node", clothes_node),
        ("Hotel_Node", hotel_node), 
        ("Restaurant_Node", restaurant_node), 
        ("Itinerary_Node", itinerary_node),
        ("Final_Summary_Node", final_summary_node),
    ]
    
    for name, fn in nodes:
        graph.add_node(name, fn)
    
    graph.set_entry_point("User_Input")

    # ---------------- Create Edges to link Nodes ---------------------
    
    # 1. Sequential Flow (Setup & IATA Resolution)
    graph.add_edge("User_Input", "Country_Recommendation")
    graph.add_edge("Country_Recommendation", "IATA_Resolver")
    
    
    # Destination IATA -> Origin IATA Resolver
    graph.add_edge("IATA_Resolver", "Origin_IATA_Resolver") 
    # Origin IATA Resolver -> Travel Cost
    graph.add_edge("Origin_IATA_Resolver", "Travel_Cost_Node") 
    
    # 2. Conditional Routing: The Budget Reset Loop
    def route_budget_check(state: TravelState):
        """Maps the string output from budget_check_node to the next node name."""
        # 
        if state["next_step"] == "Country_Recommendation_Retry":
             # This section handles the state cleanup and retry
             state["budget_adjustments"] = state.get("budget_adjustments", 0) + 1
             state["country"] = None
             state["iata_code"] = None
             state["origin_iata_code"] = None
             return "Country_Recommendation"
        
        return state["next_step"] # "Weather_Node" for success

    graph.add_conditional_edges(
        "Budget_Check_Node",
        budget_check_node, 
        {
            "Country_Recommendation_Retry": "Country_Recommendation", # IF budget is exceeded -> Full Reset
            "Weather_Node": "Weather_Node"                           # IF budget is OK -> Success Path
        }
    )

    # 3. Sequential Flow (Success Path)
    graph.add_edge("Weather_Node", "Clothes_Node")
    graph.add_edge("Clothes_Node", "Hotel_Node")
    graph.add_edge("Hotel_Node", "Restaurant_Node")
    graph.add_edge("Restaurant_Node", "Itinerary_Node")
    graph.add_edge("Itinerary_Node", "Final_Summary_Node")
    
    # 4. Final Handoff
    graph.add_edge("Final_Summary_Node", END)

    # Compile workflow
    return graph.compile()
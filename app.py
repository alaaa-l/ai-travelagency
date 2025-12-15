from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from travel_workflow import get_travel_workflow_app
from rag import get_rag_collection
from langchain_core.messages import HumanMessage, AIMessage


st.set_page_config(page_title="AI Travel Planner", layout="wide")
st.title("🌍 AI Travel Planner")

# Initialize workflow app and RAG collection in session state
if "workflow_app" not in st.session_state:
    
    st.session_state.workflow_app = get_travel_workflow_app()

if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = get_rag_collection()

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Your Travel Preferences")

budget = float(st.sidebar.text_input("Budget (USD)", "1500"))
interests = [i.strip() for i in st.sidebar.text_input(
    "Interests (comma-separated)", "beaches, nightlife"
).split(",")]
previous_destinations = [i.strip() for i in st.sidebar.text_input(
    "Previous Destinations", "France, Turkey"
).split(",")]
duration = st.sidebar.number_input("Trip duration (days)", min_value=1, max_value=30, value=7)
origin = st.sidebar.text_input("Origin country/city", "Beirut") 
date = st.sidebar.date_input("Travel date")

# ---------------- Start Button ----------------
if st.button("Plan My Trip"):

    user_info = {
        "budget": budget,
        "interests": interests,
        "previous_destinations": previous_destinations,
        "duration": duration,
        "origin": origin,
        "date": date.strftime("%Y-%m-%d"),
    }

   
    state = {
        "user_info": user_info,
        
        "messages": [HumanMessage(content="Please start planning my trip based on the provided preferences.")],
       
        "rag_collection": st.session_state.rag_collection,
        "budget_adjustments": 0,
    }

    st.subheader("📝 Travel Planning Steps")

    
    output_box = st.container() 
    all_messages = []

    # Stream the workflow execution
    for step in st.session_state.workflow_app.stream(
        state,
        {"recursion_limit": 10}
    ):
        if "messages" not in step:
            continue
        
        # Collect and format all new messages from the step
        for msg in step["messages"]:
            if isinstance(msg, HumanMessage):
                all_messages.append(f"**User:** {msg.content}")
            elif isinstance(msg, AIMessage):
                all_messages.append(f"**AI:** {msg.content}")
        
        # Update the display with the cumulative history
        with output_box:
            # Clear previous content and display the updated list
            output_box.empty()
            st.markdown("\n\n".join(all_messages))


    # ---------------- Final Output ----------------
    st.success("✅ Travel Plan Generated Successfully!")

    if all_messages:
        st.markdown("### ✨ Final Travel Summary")
        st.markdown(all_messages[-1])
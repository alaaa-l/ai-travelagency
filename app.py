from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from travel_workflow import get_travel_workflow_app

workflow_app = get_travel_workflow_app()

st.set_page_config(page_title="AI Travel Planner", layout="wide")
st.title("🌍 AI Travel Planner")

# ---------------- Step 1: Collect user inputs ----------------
st.sidebar.header("Your Travel Preferences")

budget = st.sidebar.text_input("Budget (USD)", "1500")
interests = st.sidebar.text_input("Interests (comma-separated)", "beaches, nightlife")
previous_destinations = st.sidebar.text_input("Previous Destinations", "France, Turkey")
duration = st.sidebar.number_input("Trip duration (days)", min_value=1, max_value=30, value=7)
origin = st.sidebar.text_input("Origin city", "Beirut")
date = st.sidebar.date_input("Travel date")

# ---------------- Step 2: Button to Start Workflow ----------------
if st.button("Plan My Trip"):
    user_info = {
        "budget": budget,
        "interests": interests,
        "previous_destinations": previous_destinations,
        "duration": duration,
        "origin": origin,
        "date": date.strftime("%Y-%m-%d")
    }

    # Initial state
    inputs = {"user_info": user_info, "messages": []}

    st.subheader("📝 Travel Planning Steps:")

    # Stream through workflow steps
    for step_output in workflow_app.stream(inputs, {"recursion_limit": 10}):
        # Extract messages from Annotated channel
        messages = step_output.get("messages", [])
        for msg in messages:
            if hasattr(msg, "content"):
                st.text(msg.content)

    st.success("✅ Travel Plan Generated Successfully!")

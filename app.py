import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Step 1: Define state schema
class State(TypedDict):
    mood: int
    sleep_hours: float
    break_hours: float
    risk_level: str
    recommendations: str
    affirmation: str

# Step 2: Burnout evaluation function
def evaluate_burnout_risk(mood: int, sleep: float, breaks: float) -> str:
    risk_flags = 0
    if mood <= 3:
        risk_flags += 1
    if sleep < 6:
        risk_flags += 1
    if breaks < 1:
        risk_flags += 1
    return "high" if risk_flags >= 2 else "low"

# Step 3: Streamlit UI
st.set_page_config(page_title="Self-Care Burnout Assistant", layout="centered")
st.title("🧘 Self-Care Scheduler AI for Burnout Prevention")

st.markdown("This AI agent gives gentle, personalized suggestions based on your mood, sleep, and breaks.")

# Input Gemini API Key
api_key = st.text_input("🔑 Enter your Gemini API key", type="password")

# Input mood, sleep, breaks
mood = st.slider("😌 How’s your mood today? (1 = bad, 5 = great)", 1, 5, 3)
sleep_hours = st.slider("🛏️ How many hours did you sleep last night?", 0.0, 12.0, 6.0, step=0.5)
break_hours = st.slider("☕ How many hours did you take breaks today?", 0.0, 5.0, 1.0, step=0.5)

submit = st.button("🧾 Get Self-Care Advice")

if submit:
    if not api_key:
        st.error("Please enter your Gemini API key.")
    else:
        # Step 4: Initialize Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=api_key)

        # Step 5: Define LangGraph nodes
        def start_node(state: State) -> State:
            return {"mood": mood, "sleep_hours": sleep_hours, "break_hours": break_hours}

        def analyzer_node(state: State) -> State:
            risk = evaluate_burnout_risk(state["mood"], state["sleep_hours"], state["break_hours"])
            return {"risk_level": risk}

        def recommendation_node(state: State) -> State:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You're a self-care advisor for university students."),
                ("human", f"""Suggest 3 simple, practical self-care activities for a student who is showing signs of burnout:
                  Mood rating: {state['mood']}
                  Sleep hours: {state['sleep_hours']}
                  Breaks today: {state['break_hours']}
                  Be gentle, specific, and encouraging.""")
            ])
            response = (prompt | llm).invoke({})
            return {"recommendations": response.content.strip()}

        def affirmation_node(state: State) -> State:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You're a motivational self-care coach."),
                ("human", "The student is doing well and not at burnout risk. Write a short, encouraging affirmation.")
            ])
            response = (prompt | llm).invoke({})
            return {"affirmation": response.content.strip()}

        def router_node(state: State) -> str:
            return "recommendation" if state["risk_level"] == "high" else "affirmation"

        # Step 6: Build LangGraph
        builder = StateGraph(State)
        builder.add_node("start", start_node)
        builder.add_node("analyzer", analyzer_node)
        builder.add_node("recommendation", recommendation_node)
        builder.add_node("affirmation", affirmation_node)

        builder.set_entry_point("start")
        builder.add_edge("start", "analyzer")
        builder.add_conditional_edges("analyzer", router_node, {
            "recommendation": "recommendation",
            "affirmation": "affirmation"
        })
        builder.add_edge("recommendation", END)
        builder.add_edge("affirmation", END)

        app_graph = builder.compile()

        # Step 7: Execute with single invoke (only 1 Gemini call made)
        final_state = app_graph.invoke({})

        # Step 8: Show output
        st.markdown("### 📊 Burnout Risk Level")
        st.success(final_state.get("risk_level", "unknown").upper())

        if final_state.get("recommendations"):
            st.markdown("### 🧘 Self-Care Recommendations")
            st.info(final_state["recommendations"])
        elif final_state.get("affirmation"):
            st.markdown("### 🌟 Affirmation")
            st.info(final_state["affirmation"])


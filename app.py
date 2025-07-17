import streamlit as st
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Title
st.set_page_config(page_title="AI for Burnout Prevention", page_icon="💆")
st.title("💆‍♀️ Self-Care AI Assistant for Burnout Prevention")

# Input Gemini API Key
api_key = st.text_input("🔐 Enter your Gemini API Key", type="password")
if not api_key:
    st.warning("Please enter your Gemini API key to begin.")
    st.stop()

# Set API Key
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

# State definition
class State(TypedDict):
    mood: int
    sleep_hours: float
    break_hours: float
    risk_level: str
    recommendations: str
    affirmation: str

# Evaluate burnout logic
def evaluate_burnout_risk(mood: int, sleep: float, breaks: float) -> str:
    risk_flags = 0
    if mood <= 3: risk_flags += 1
    if sleep < 6: risk_flags += 1
    if breaks < 1: risk_flags += 1
    return "high" if risk_flags >= 2 else "low"

# Start node
def start_node(state: State) -> State:
    return {
        "mood": state["mood"],
        "sleep_hours": state["sleep_hours"],
        "break_hours": state["break_hours"]
    }

# Analyzer node
def analyzer_node(state: State) -> State:
    risk = evaluate_burnout_risk(state["mood"], state["sleep_hours"], state["break_hours"])
    return {"risk_level": risk}

# Recommendation node
def recommendation_node(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a self-care advisor for university students."),
        ("human", f"""Suggest 3 gentle, specific self-care activities for a student feeling burnout:
        Mood: {state['mood']}
        Sleep hours: {state['sleep_hours']}
        Breaks today: {state['break_hours']}""")
    ])
    chain = prompt | llm
    response = chain.invoke({})
    return {"recommendations": response.content.strip()}

# Affirmation node
def affirmation_node(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a motivational self-care coach."),
        ("human", "Write a short, encouraging affirmation for a student not at burnout risk.")
    ])
    chain = prompt | llm
    response = chain.invoke({})
    return {"affirmation": response.content.strip()}

# Router
def router_node(state: State) -> str:
    return "recommendation" if state["risk_level"] == "high" else "affirmation"

# Build LangGraph
builder = StateGraph(State)
builder.add_node("start", start_node)
builder.add_node("analyzer", analyzer_node)
builder.add_node("recommendation", recommendation_node)
builder.add_node("affirmation", affirmation_node)

builder.set_entry_point("start")
builder.add_edge("start", "analyzer")
builder.add_conditional_edges(
    "analyzer", router_node, {
        "recommendation": "recommendation",
        "affirmation": "affirmation"
    }
)
builder.add_edge("recommendation", END)
builder.add_edge("affirmation", END)

graph_app = builder.compile()

# Streamlit Inputs
with st.form("self_care_form"):
    mood = st.slider("😊 Rate your mood today (1 = Bad, 5 = Great)", 1, 5, 3)
    sleep = st.number_input("🛌 Hours you slept last night", min_value=0.0, max_value=24.0, value=7.0)
    breaks = st.number_input("☕ Break time taken today (hours)", min_value=0.0, max_value=24.0, value=2.0)
    submitted = st.form_submit_button("Get Self-Care Guidance")

if submitted:
    # Run the agent
    state = {
        "mood": mood,
        "sleep_hours": sleep,
        "break_hours": breaks
    }
    result = graph_app.invoke(state)

    st.subheader("📊 Burnout Risk Level:")
    st.info(result["risk_level"].upper())

    if result["risk_level"] == "high":
        st.subheader("🧘 Self-Care Recommendations:")
        st.success(result["recommendations"])
    else:
        st.subheader("🌟 Affirmation:")
        st.success(result["affirmation"])

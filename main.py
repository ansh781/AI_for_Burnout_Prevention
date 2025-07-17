
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize Gemini LLM
import os
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

# Burnout evaluation function
def evaluate_burnout_risk(mood, sleep, breaks):
    risk_flags = 0
    if mood <= 3:
        risk_flags += 1
    if sleep < 6:
        risk_flags += 1
    if breaks < 1:
        risk_flags += 1
    return "high" if risk_flags >= 2 else "low"

# Self-care recommendations using Gemini
def get_recommendations(mood, sleep, breaks):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a self-care advisor for university students."),
        ("human", f"""Suggest 3 simple, practical self-care activities for a student showing signs of burnout:
Mood rating: {mood}
Sleep hours: {sleep}
Break hours: {breaks}
Be gentle, specific, and encouraging.""")
    ])
    chain = prompt | llm
    return chain.invoke({}).content.strip()

# Affirmation if not at risk
def get_affirmation():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a motivational self-care coach."),
        ("human", "The student is doing well and not at burnout risk. Write a short, encouraging affirmation.")
    ])
    chain = prompt | llm
    return chain.invoke({}).content.strip()

# --- Streamlit UI ---
st.title("🧘 Self-Care AI for Burnout Prevention")

mood = st.slider("How would you rate your mood today? (1 = bad, 5 = great)", 1, 5, 3)
sleep = st.number_input("How many hours did you sleep last night?", min_value=0.0, max_value=24.0, value=7.0)
breaks = st.number_input("How many hours did you take proper breaks today?", min_value=0.0, max_value=24.0, value=2.0)

if st.button("Submit"):
    risk = evaluate_burnout_risk(mood, sleep, breaks)
    st.markdown(f"### 📊 Burnout Risk Level: `{risk.upper()}`")
    
    if risk == "high":
        recs = get_recommendations(mood, sleep, breaks)
        st.markdown("### 🧘 Self-Care Recommendations:")
        st.write(recs)
    else:
        affirm = get_affirmation()
        st.markdown("### 🌟 Affirmation:")
        st.write(affirm)

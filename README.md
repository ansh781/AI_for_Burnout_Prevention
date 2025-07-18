# Self-Care Scheduler AI 🤖🧘

A burnout prevention assistant built with LangGraph, Gemini API, and Streamlit.

## 💡 How it Works

This AI assistant helps university students track and manage burnout risk through a simple 4-step process:

1. **User Inputs** 🧠  
   You rate your **mood**, **sleep hours**, and **break time** using sliders.

2. **Burnout Risk Analysis** 📊  
   Based on your input, the app checks if you're at **high** or **low** burnout risk.

3. **Self-Care Guidance** 💌  
   - If you're at **high risk**, it suggests 3 personalized self-care tips using the Gemini AI model.  
   - If you're at **low risk**, it gives you a positive affirmation to stay motivated.

4. **Built with LangGraph + Gemini API** 🤖  
   The app uses LangChain’s LangGraph to structure this decision-making flow and only makes **one smart API call** to Gemini.


import google.generativeai as genai

genai.configure(api_key="AIzaSyBxJfg5ly3nL-Feer4dAh0jY8oO5-iPIX8")

def optimization_agent(state):
    metrics = state["metrics"]
    target = state["target"]
    logs = state.get("logs", [])

    prompt = f"""
    You are a model optimization AI agent. Here are the model metrics and issue:
    - Metrics: {metrics}
    Suggest the best optimization or hyperparameter tuning steps for the current model (code in Python/sklearn if possible).
    Respond in exactly two lines, covering all key info. Do not write a long paragraph.
    """

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    suggestion = response.text
    logs.append("Suggestion for optimization :\n" + suggestion)

    return {
        "df": state["df"],
        "target": target,
        "model": state["model"],
        "metrics": metrics,
        "logs": logs
    }

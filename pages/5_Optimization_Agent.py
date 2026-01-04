import streamlit as st
import pandas as pd
from agents.optimization_Agent import optimization_agent

def get_df():
    if "wizard_df" not in st.session_state:
        st.session_state.wizard_df = pd.read_csv("data/titanic.csv")
    return st.session_state.wizard_df.copy()

def get_logs():
    if "wizard_logs" not in st.session_state:
        st.session_state.wizard_logs = []
    return st.session_state.wizard_logs

def set_df(df):
    st.session_state.wizard_df = df.copy()

def set_logs(logs):
    st.session_state.wizard_logs = logs.copy()

def set_metrics(metrics):
    st.session_state.wizard_metrics = metrics

st.header("Optimization Agent")

if st.button("Run Optimization Agent"):
    state = {"df": get_df(), "target": "Survived", "model": st.session_state.get("wizard_model", None), "metrics": st.session_state.get("wizard_metrics", {}), "logs": get_logs()}
    out = optimization_agent(state)
    set_df(out["df"])
    set_logs(out["logs"])
    set_metrics(out["metrics"])
    st.success("Optimization completed!")

st.dataframe(get_df())
if get_logs():
    last_log = get_logs()[-1]
    if "Suggestion for optimization" in last_log:
        suggestion = last_log.split("Suggestion for optimization :",1)[-1].strip()
        st.markdown(f"""
    <div style='background: #f7cac9; border-radius: 8px; padding: 12px 18px; margin-bottom: 10px; color: #222; font-size: 1.1em; font-family: Georgia,serif; box-shadow: 0 2px 8px #0001;'>
    <b>Optimization Suggestion:</b><br>
    <span style='color:#0b5394; font-weight:bold;'>{suggestion}</span>
    </div>
    """, unsafe_allow_html=True)
    else:
        st.info(last_log)
if "wizard_metrics" in st.session_state:
    st.write("**Metrics:**", st.session_state.wizard_metrics)

if st.button("Restart Wizard"):
    st.session_state.wizard_df = pd.read_csv("data/titanic.csv")
    st.session_state.wizard_logs = []
    st.session_state.wizard_model = None
    st.session_state.wizard_metrics = None
    st.switch_page("pages/1_Data_Quality_Agent.py")

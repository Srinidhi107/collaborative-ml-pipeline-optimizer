import streamlit as st
import pandas as pd
from agents.debug_agent import debug_agent

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

def set_model(model):
    st.session_state.wizard_model = model

def set_metrics(metrics):
    st.session_state.wizard_metrics = metrics

st.header("Debug Agent")

if st.button("Run Debug Agent"):
    state = {"df": get_df(), "target": "Survived", "model": st.session_state.get("wizard_model", None), "metrics": st.session_state.get("wizard_metrics", {}), "logs": get_logs()}
    out = debug_agent(state)
    set_df(out["df"])
    set_logs(out["logs"])
    set_model(out.get("model", None))
    set_metrics(out["metrics"])
    st.success("Debugging completed!")

st.dataframe(get_df())
if get_logs():
    st.info(get_logs()[-1])
if "wizard_metrics" in st.session_state:
    st.write("**Metrics:**", st.session_state.wizard_metrics)

if st.button("Next: Optimization Agent"):
    st.switch_page("pages/5_Optimization_Agent.py")

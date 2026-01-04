import streamlit as st
import pandas as pd
from agents.feature_engineer_agent import feature_agent

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

st.header("Feature Engineering Agent")

if st.button("Run Feature Engineering Agent"):
    state = {"df": get_df(), "target": "Survived", "logs": get_logs()}
    out = feature_agent(state)
    set_df(out["df"])
    set_logs(out["logs"])
    st.success("Feature Engineering completed!")

st.dataframe(get_df())
if get_logs():
    st.info(get_logs()[-1])

if st.button("Next: Train Model Agent"):
    st.switch_page("pages/3_Train_Model_Agent.py")

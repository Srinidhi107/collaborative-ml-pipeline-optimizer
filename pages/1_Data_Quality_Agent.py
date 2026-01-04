import streamlit as st
import pandas as pd
from agents.data_quality_agent import data_quality_agent

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

st.header("Data Quality Agent")

if st.button("Run Data Quality Agent"):
    state = {"df": get_df(), "target": "Survived", "logs": get_logs()}
    out = data_quality_agent(state)
    set_df(out["df"])
    set_logs(out["logs"])
    st.success("Data Quality Agent completed!")

st.dataframe(get_df())
if get_logs():
    st.info(get_logs()[-1])

if st.button("Next: Feature Engineering Agent"):
    st.switch_page("pages/2_Feature_Engineering_Agent.py")

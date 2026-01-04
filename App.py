import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Multi-Agent ML Pipeline", layout="wide")

st.title("Collaborative Multi-Agent ML Pipeline Optimizer")


# =========================
# SECTION 0: DATASET SELECTION
# =========================
st.write("Get started by uploading your dataset and specifying the target column.")

uploaded_file = st.file_uploader("Upload the dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset uploaded! Please select your target column below.")
else:
    df = pd.read_csv("data/titanic.csv")
    st.write("Using the default Titanic dataset.")

# Target selection with chatbot feedback (manual input)
TARGET = st.text_input("Enter the target column name (case-sensitive):", "")
if TARGET:
    st.write(f"Target column set to: {TARGET}")
else:
    st.write("❗Please enter the target column name to proceed.")

# Main landing page for wizard navigation
st.header("Live Automation")
st.markdown("""
Welcome! Click below to start the live, step-by-step agent pipeline.
""")

if st.button("Start Live Automation"):
    st.switch_page("pages/1_Data_Quality_Agent.py")

# Load results
results = pd.read_csv("results.csv", index_col=0)

baseline = results["baseline"].copy()
optimized = results["optimized"].copy()

# Convert numeric values safely
for key in ["train_accuracy", "test_accuracy"]:
    baseline[key] = float(baseline[key])
    optimized[key] = float(optimized[key])


# =========================
# SECTION 3: BAR CHART
# =========================
st.subheader("Baseline vs Optimized Accuracy")

col1, col2 = st.columns([1, 2])
with col1:
    fig, ax = plt.subplots(figsize=(3,2))  # Reduced size: width=2, height=1 inches
    labels = ["Train Accuracy", "Test Accuracy"]
    baseline_vals = [
        baseline["train_accuracy"],
        baseline["test_accuracy"]
    ]
    optimized_vals = [
        optimized["train_accuracy"],
        optimized["test_accuracy"]
    ]
    x = range(len(labels))
    ax.bar(x, baseline_vals, width=0.2, label="Baseline")
    ax.bar([i + 0.2 for i in x], optimized_vals, width=0.2, label="Optimized")
    ax.set_xticks([i + 0.1 for i in x])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(prop={'size': 6})
    st.pyplot(fig)
with col2:
    st.write("")  # Placeholder to avoid IndentationError


# After running the Data Quality Agent, display the LLM's suggestion in the UI
if "logs" in st.session_state and st.session_state.logs:
    last_log = st.session_state.logs[-1]
    if "LLM DataQualityAgent suggestion" in last_log:
        st.markdown("**LLM Data Quality Agent Suggestion:**")
        st.code(last_log.split('LLM DataQualityAgent suggestion:')[-1].strip(), language="markdown")

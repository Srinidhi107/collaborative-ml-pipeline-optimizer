import pandas as pd
from pipelines.baselinemodel import run_baseline
from graphs.pipeline_graphs import build_graph

# Load dataset
df = pd.read_csv("data/titanic.csv")

TARGET = "Survived"

# =========================
# BASELINE PIPELINE
# =========================
baseline_metrics = run_baseline(df, TARGET)

# =========================
# AGENT PIPELINE (LangGraph)
# =========================
intermediate_datasets = {}
intermediate_logs = []

# Step 1: Data Quality Agent
from agents.data_quality_agent import data_quality_agent
state1 = {
    "df": df.copy(),
    "target": TARGET,
    "logs": []
}
out1 = data_quality_agent(state1)
intermediate_datasets["Data Quality"] = out1["df"].copy()
intermediate_logs.append(out1["logs"][-1])

# Step 2: Feature Engineering Agent
from agents.feature_engineer_agent import feature_agent
state2 = {
    "df": out1["df"].copy(),
    "target": TARGET,
    "logs": out1["logs"]
}
out2 = feature_agent(state2)
intermediate_datasets["Feature Engineering"] = out2["df"].copy()
intermediate_logs.append(out2["logs"][-1])

# Step 3: Train Model
from pipelines.train_model import train_model
state3 = {
    "df": out2["df"].copy(),
    "target": TARGET,
    "logs": out2["logs"]
}
out3 = train_model(state3)
intermediate_datasets["Train Model"] = out3["df"].copy()
intermediate_logs.append(out3["logs"][-1])

# Step 4: Debug Agent
from agents.debug_agent import debug_agent
state4 = {
    "df": out3["df"].copy(),
    "target": TARGET,
    "model": out3["model"],
    "metrics": out3["metrics"],
    "logs": out3["logs"]
}
out4 = debug_agent(state4)
intermediate_datasets["Debug"] = out4["df"].copy()
intermediate_logs.append(out4["logs"][-1])

# Step 5: Optimization Agent
from agents.optimization_Agent import optimization_agent
state5 = {
    "df": out4["df"].copy(),
    "target": TARGET,
    "model": out4["model"],
    "metrics": out4["metrics"],
    "logs": out4["logs"]
}
out5 = optimization_agent(state5)
intermediate_datasets["Optimization"] = out5["df"].copy()
intermediate_logs.append(out5["logs"][-1])

optimized_metrics = out5["metrics"]

# =========================
# PRINT RESULTS
# =========================
print("\n=== BASELINE METRICS ===")
print(baseline_metrics)

print("\n=== OPTIMIZED METRICS ===")
print(optimized_metrics)

print("\n=== AGENT LOGS ===")
for log in intermediate_logs:
    print(log)

# =========================
# SAVE RESULTS FOR STREAMLIT
# =========================

results = {
    "baseline": baseline_metrics,
    "optimized": optimized_metrics
}
pd.DataFrame(results).to_csv("results.csv")

# Save intermediate datasets for Streamlit
for step, data in intermediate_datasets.items():
    data.to_csv(f"intermediate_{step.replace(' ', '_').lower()}.csv", index=False)

# Save logs for Streamlit
pd.DataFrame({"step": list(intermediate_datasets.keys()), "log": intermediate_logs}).to_csv("intermediate_logs.csv", index=False)

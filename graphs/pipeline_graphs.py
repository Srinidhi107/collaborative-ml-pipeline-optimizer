from langgraph.graph import StateGraph

from agents.data_quality_agent import data_quality_agent
from agents.feature_engineer_agent import feature_agent
from pipelines.train_model import train_model
from agents.debug_agent import debug_agent
from agents.optimization_Agent import optimization_agent

def build_graph():
    graph = StateGraph(dict)

    graph.add_node("data_quality", data_quality_agent)
    graph.add_node("feature", feature_agent)
    graph.add_node("train", train_model)
    graph.add_node("debug", debug_agent)
    graph.add_node("optimize", optimization_agent)

    graph.set_entry_point("data_quality")

    graph.add_edge("data_quality", "feature")
    graph.add_edge("feature", "train")
    graph.add_edge("train", "debug")
    graph.add_edge("debug", "optimize")

    return graph.compile()

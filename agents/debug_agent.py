def debug_agent(state):
    metrics = state["metrics"]
    target = state["target"]

    train_acc = metrics["train_accuracy"]
    test_acc = metrics["test_accuracy"]

    if train_acc - test_acc > 0.1:
        issue = "Overfitting"
    elif train_acc < 0.6:
        issue = "Underfitting"
    else:
        issue = "Good Fit"

    metrics["issue"] = issue
    state["logs"].append(f"DebugAgent: {issue}")

    return {
        "df": state["df"],
        "target": target,
        "model": state["model"],
        "metrics": metrics,
        "logs": state["logs"]
    }

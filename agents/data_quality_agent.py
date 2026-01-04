def data_quality_agent(state):
    df = state["df"].copy()
    target = state["target"]
    logs = state.get("logs", [])

    # Simple static data quality logic
    summary = df.isnull().sum().to_dict()
    types = df.dtypes.astype(str).to_dict()
    suggestion = []
    for col, missing in summary.items():
        if missing > 0:
            if types[col] == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
                suggestion.append(f"Filled missing values in {col} with mode.")
            else:
                df[col] = df[col].fillna(df[col].mean())
                suggestion.append(f"Filled missing values in {col} with mean.")
    logs.append("DataQualityAgent actions:\n" + "\n".join(suggestion))

    return {
        "df": df,
        "target": target,
        "logs": logs
    }
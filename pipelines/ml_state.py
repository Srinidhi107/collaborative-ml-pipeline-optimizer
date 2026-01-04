def create_initial_state(df, target_column):
    return {
        "df": df,
        "target": target_column,
        "model": None,
        "metrics": {},
        "logs": []
    }

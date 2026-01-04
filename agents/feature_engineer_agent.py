
from sklearn.preprocessing import LabelEncoder, StandardScaler

def feature_agent(state):
    df = state["df"].copy()
    target = state["target"]

    # Encode categorical columns
    for col in df.columns:
        if df[col].dtype == "object" and col != target:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Scale numeric features
    scaler = StandardScaler()
    feature_cols = df.drop(target, axis=1).columns
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    state["logs"].append("FeatureAgent: encoding & scaling applied")

    return {
        "df": df,
        "target": target,
        "logs": state["logs"]
    }

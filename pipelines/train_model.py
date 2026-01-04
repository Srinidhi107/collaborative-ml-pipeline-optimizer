from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(state):
    df = state["df"]
    target = state["target"]

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))

    state["logs"].append(
        f"TrainingNode: train_acc={train_acc:.2f}, test_acc={test_acc:.2f}"
    )

    return {
        "df": df,
        "target": target,
        "model": model,
        "metrics": {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
        },
        "logs": state["logs"]
    }

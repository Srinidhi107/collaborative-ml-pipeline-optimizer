from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pandas as pd

def run_baseline(df, target):
    df = df.copy()

    # Encode categorical columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(target, axis=1)
    y = df[target]

    # BASIC imputation (mean)
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return {
        "train_accuracy": model.score(X_train, y_train),
        "test_accuracy": accuracy_score(y_test, model.predict(X_test))
    }

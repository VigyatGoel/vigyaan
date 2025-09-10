import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def train_models(X, y, save_path: str = "models/") -> Dict[str, Dict[str, float]]:
    """_summary_

    Returns:
        _type_: _description_
    """
    # Ensure save directory exists
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(),
        "SVC": SVC(probability=True),
        "RandomForestClassifier": RandomForestClassifier(),
    }

    # Train and evaluate models
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        # For classification problems
        try:
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        except Exception:
            precision = recall = f1 = 0.0

        # MSE (useful for regression-like evaluation)
        try:
            mse = mean_squared_error(y_test, y_pred)
        except Exception:
            mse = 0.0

        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mse": mse,
        }

        # Save model
        with open(f"{save_path}{model_name}.pkl", "wb") as file:
            pickle.dump(model, file)

    return results


def predict_with_model(
    model_path: str, features: np.ndarray
) -> Tuple[Any, List[float]]:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    prediction = model.predict(features)

    # Check if model has predict_proba method
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0].tolist()
    else:
        probabilities = []

    return prediction[0], probabilities

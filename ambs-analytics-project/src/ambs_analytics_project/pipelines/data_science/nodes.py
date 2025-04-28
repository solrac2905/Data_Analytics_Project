from typing import Tuple
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def train_test_split_node(
    df_final: pd.DataFrame,
    target_col: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets.

    Args:
        df_final (pd.DataFrame): The input DataFrame containing features and target.
        target_col (str): The name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            - X_train: Training features
            - X_test: Testing features
            - y_train: Training target
            - y_test: Testing target
    """

    X = df_final.drop(columns=target_col)  # Features
    y = df_final[target_col]  # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def logistic_regression_node(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    solver: str = "liblinear",
    class_weight: str = "balanced",
    max_iter: int = 10000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Trains a logistic regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

        solver (str): Algorithm to use in the optimization problem.
        class_weight (str): Weights associated with classes.
        max_iter (int): Maximum number of iterations.
        random_state (int): Random state for reproducibility.

    Returns:
        LogisticRegression: Trained logistic regression model.
    """

    model = LogisticRegression(
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def roc_auc_node(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, float]:
    """
    Computes ROC AUC scores for training and testing sets.

    Args:
        model (LogisticRegression): Trained logistic regression model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): True labels for training set.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): True labels for testing set.

    Returns:
        Tuple[float, float]:
            - auc_train: ROC AUC score for training set.
            - auc_test: ROC AUC score for testing set.
    """
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, y_train_proba)
    auc_test = roc_auc_score(y_test, y_test_proba)

    logger.debug(f"TRAIN Model AUC: {auc_train}")
    logger.debug(f"TEST Model AUC: {auc_test}")

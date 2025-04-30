from typing import Tuple, Dict
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
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
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> LogisticRegression:
    """
    Tune a logistic regression model using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        LogisticRegression: Best logistic regression model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 10)
    max_iter = params.get("max_iter", 10000)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)

    # Step 1: Initialize logistic regression
    logreg_base = LogisticRegression(max_iter=max_iter, random_state=random_state)

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=logreg_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    # Best logistic regression model
    model = grid_search.best_estimator_

    return model


def decision_tree_classifier_node(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> DecisionTreeClassifier:
    """
    Tune a Decision Tree Classifier using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        DecisionTreeClassifier: Best decision tree model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 10)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)

    # Step 1: Initialize decision tree classifier
    dt_base = DecisionTreeClassifier(random_state=random_state)

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=dt_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    # Best decision tree model
    model = grid_search.best_estimator_

    return model


def random_forest_classifier_node(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> RandomForestClassifier:
    """
    Tune a Random Forest Classifier using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        RandomForestClassifier: Best random forest model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 10)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)

    # Step 1: Initialize Random Forest Classifier
    rf_base = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    # Best Random Forest model
    model = grid_search.best_estimator_

    return model

#CORREGIR SCALE POST WEIGHT
def xgboost_classifier_node(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> XGBClassifier:
    """
    Tune an XGBoost Classifier using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        XGBClassifier: Best XGBoost model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 10)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)

    # Step 1: Initialize XGBoost Classifier with imbalance handling
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(
        y_train
    )  # Balance class weights
    
    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    # Best XGBoost model
    model = grid_search.best_estimator_

    return model


def neural_network_classifier_node(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> MLPClassifier:
    """
    Tune an MLPClassifier (Neural Network) using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        MLPClassifier: Best MLP model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 10)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)
    max_iter = params.get("max_iter", 1000)

    # Step 1: Initialize MLP Classifier
    mlp_base = MLPClassifier(max_iter=max_iter, random_state=random_state)

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=mlp_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    # Best MLP model
    model = grid_search.best_estimator_

    return model


def lightgbm_classifier_node(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> LGBMClassifier:
    """
    Tune a LightGBM Classifier using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        LGBMClassifier: Best LightGBM model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 5)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)

    # Step 1: Initialize LightGBM Classifier
    lgbm_base = LGBMClassifier(random_state=random_state)

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=lgbm_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    # Best LightGBM model
    model = grid_search.best_estimator_

    return model


def catboost_classifier_node(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> CatBoostClassifier:
    """
    Tune a CatBoost Classifier using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        CatBoostClassifier: Best CatBoost model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 5)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)

    # Step 1: Initialize CatBoost Classifier
    cb_base = CatBoostClassifier(
        verbose=0, random_state=random_state  # Suppress CatBoost training logs
    )

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=cb_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    # Best CatBoost model
    model = grid_search.best_estimator_

    return model


def results(
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

    y_test_pred = model.predict(X_test)

    auc_train = roc_auc_score(y_train, y_train_proba)
    auc_test = roc_auc_score(y_test, y_test_proba)

    logger.info(f"Model: {type(model).__name__}")
    logger.info(f"TRAIN Model AUC: {auc_train:.4f}")
    logger.info(f"TEST Model AUC: {auc_test:.4f}")

    TN_logistic, FP_logistic, FN_logistic, TP_logistic = confusion_matrix(
        y_test, y_test_pred
    ).ravel()
    logger.info(f"True-Negatives (TN): {TN_logistic}")
    logger.info(f"False-Positives (FP): {FP_logistic}")
    logger.info(f"False-Negatives (FN): {FN_logistic}")
    logger.info(f"True-Positives (TP): {TP_logistic}")
    exp_profit_logistic = TN_logistic - FN_logistic - FP_logistic
    logger.info(f"Expected Profit: {exp_profit_logistic}")

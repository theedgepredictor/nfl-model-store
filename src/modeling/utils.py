import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

import joblib
import mlflow
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from src.data_registry import DataRegistry


@dataclass
class ModelConfig:
    experiment_name: str
    run_name: str
    metric: str
    metric_threshold: float
    run_season: int
    run_week: Optional[int]
    feature_store_group: str
    feature_store_name: str
    data_features: List[str]
    model_class: str
    start_season: int = 2002
    run_date: str = datetime.now().strftime("%Y-%m-%d")
    metadata_cols: Optional[List[str]] = None

    def to_dict(self):
        """Converts the dataclass attributes to a dictionary."""
        return asdict(self)


def train_test_splitter(df, data_features, start_season, run_season, run_week=None, metadata_cols=None, include_extra_week = False):
    target = data_features[0]
    features = data_features[1:]

    # Default metadata columns if none are provided
    if metadata_cols is None:
        metadata_cols = ['home_team', 'away_team', 'season', 'week', target]

    # If run_week is specified, train up to the current week in the run_season
    if run_week is not None:
        X_train = df.loc[
            (df['season'] >= start_season) &
            ((df['season'] < run_season) |
             ((df['season'] == run_season) & (df['week'] < run_week))),
            features
        ]
        y_train = df.loc[
            (df['season'] >= start_season) &
            ((df['season'] < run_season) |
             ((df['season'] == run_season) & (df['week'] < run_week))),
            target
        ]

        # Test set is the current week and the next week (run_week and run_week + 1)
        X_test = df.loc[
            (df['season'] == run_season) &
            (df['week'].isin([run_week, run_week + 1] if include_extra_week else [run_week])),
            features
        ]
        y_test = df.loc[
            (df['season'] == run_season) &
            (df['week'].isin([run_week, run_week + 1] if include_extra_week else [run_week])),
            target
        ]

        # Metadata for test set
        metadata_test = df.loc[
            (df['season'] == run_season) &
            (df['week'].isin([run_week, run_week + 1] if include_extra_week else [run_week])),
            metadata_cols
        ]
    else:
        # If run_week is None, train on data from previous seasons up to (run_season - 1)
        X_train = df.loc[
            (df['season'] >= start_season) &
            (df['season'] < run_season),
            features
        ]
        y_train = df.loc[
            (df['season'] >= start_season) &
            (df['season'] < run_season),
            target
        ]

        # Test set is all data from the current run_season
        X_test = df.loc[df['season'] == run_season, features]
        y_test = df.loc[df['season'] == run_season, target]

        # Metadata for test set
        metadata_test = df.loc[df['season'] == run_season, metadata_cols]

    return X_train, y_train, X_test, y_test, metadata_test

##################################################
## Feature Selection
##################################################

def plot_correlation(df, data_features):
    """
    Plots the correlation of each variable in the dataframe with the 'demand' column.

    Args:
    - df (pd.DataFrame): DataFrame containing the data, including a 'demand' column.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - None (Displays the plot on a Jupyter window)
    """

    # Compute correlations between all variables and 'demand'
    features = df[data_features].copy()
    correlations = features.corr()[data_features[0]].drop(data_features[0]).sort_values()

    # Generate a color palette from red to green
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)

    # Set Seaborn style
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )  # Light grey background and thicker grid lines

    # Create bar plot
    fig = plt.figure(figsize=(12, 8))
    plt.barh(correlations.index, correlations.values, color=color_mapped)

    # Set labels and title with increased font size
    plt.title(f"Correlation with {data_features[0]}", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # prevent matplotlib from displaying the chart every time we call this function
    plt.close(fig)

    return fig

#############################################
## Feature Importance
#############################################

def plot_feature_importance(model, booster):
    """
    Plots feature importance for an XGBoost model.

    Args:
    - model: A trained XGBoost model

    Returns:
    - fig: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_type = "weight" if booster == "gblinear" else "gain"
    xgb.plot_importance(
        model,
        importance_type=importance_type,
        ax=ax,
        title=f"Feature Importance based on {importance_type}",
    )
    plt.tight_layout()
    plt.close(fig)

    return fig

def feature_importances(clf, features, name):
    """Get feature importance scores for XGBoost or RandomForest models"""

    if hasattr(clf, "get_score"):  # XGBoost feature importance extraction
        coef_ = clf.get_score(importance_type='weight')
        features_coef_sorted = sorted(coef_.items(), key=lambda x: x[1], reverse=True)

    elif hasattr(clf, "feature_importances_"):  # RandomForest feature importance extraction
        coef_ = dict(zip(features, clf.feature_importances_))
        features_coef_sorted = sorted(coef_.items(), key=lambda x: x[1], reverse=True)

    else:
        raise ValueError(f"Model {type(clf)} not supported for feature importance extraction.")

    # Extract the sorted features and their coefficients
    features_sorted = [feature for feature, _ in features_coef_sorted]
    coef_sorted = [coef for _, coef in features_coef_sorted]

    # Optionally, print or store the feature importances
    print(f"Feature importance for {name}:")
    for feature, coef in zip(features_sorted, coef_sorted):
        print(f"{feature}: {coef}")

    return {coef: feature for feature, coef in zip(features_sorted, coef_sorted)}

#############################################
## Plot Residuals
#############################################

def plot_residuals(model, dvalid, valid_y):
    """
    Plots the residuals of the model predictions against the true values.

    Args:
    - model: The trained XGBoost model.
    - dvalid (xgb.DMatrix): The validation data in XGBoost DMatrix format.
    - valid_y (pd.Series): The true values for the validation set.

    Returns:
    - None (Displays the residuals plot on a Jupyter window)
    """

    # Predict using the model
    preds = model.predict(dvalid)

    # Calculate residuals
    residuals = valid_y - preds

    # Set Seaborn style
    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    # Create scatter plot
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(valid_y, residuals, color="blue", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-")

    # Set labels, title and other plot properties
    plt.title("Residuals vs True Values", fontsize=18)
    plt.xlabel("True Values", fontsize=16)
    plt.ylabel("Residuals", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")

    plt.tight_layout()

    # Show the plot
    plt.close(fig)

    return fig


def get_predictions(experiment_name, run_name, season):
    """
    Get predictions for a specific season from the saved parquet file.

    Args:
        experiment_name (str): The name of the experiment.
        run_name (str): The name of the run.
        season (int): The season to get predictions for.

    Returns:
        pd.DataFrame or None: The predictions for the specified season, or None if not found.
    """
    try:
        # Define the path to the predictions parquet file
        file_path = f'./data/predictions/{experiment_name}/{run_name}/{season}.parquet'

        # Check if the file exists
        if os.path.exists(file_path):
            # Load the predictions from the parquet file
            predictions_df = pd.read_parquet(file_path)
            return predictions_df
        else:
            print(f"No predictions found for season {season}.")
            return None
    except Exception as e:
        print(f"Error occurred while retrieving predictions: {e}")
        return None


def load_state_config(root_path):
    """Load the hyperparameters from a JSON file if they exist."""
    filepath = f"{root_path}/config.json"
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            hyperparameters = json.load(f)
        return hyperparameters
    return None

def save_state_config(hyperparameters, root_path):
    """Save the best hyperparameters to a JSON file."""
    os.makedirs(f"{root_path}", exist_ok=True)
    with open(f"{root_path}/config.json", 'w') as f:
        json.dump(hyperparameters, f, indent=4)


def register_model(model, experiment_name, run_name):
    """
    Save the trained model to a specific directory.

    Parameters:
        model: The trained model (e.g., RandomForestClassifier).
        directory_path (str): Path to the directory where the model will be saved.
        file_name (str): The name of the file to save the model (default is 'model.pkl').

    Returns:
        str: The path where the model was saved.
    """
    # Ensure the directory exists
    output_dir = f"./src/experiments/{experiment_name}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the model as a pickle file
    model_path = os.path.join(output_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    return model_path


# Load the saved model
def load_registered_model(experiment_name, run_name):
    """
    Load a saved model from a specific directory.

    Parameters:
        directory_path (str): Path to the directory where the model is saved.
        file_name (str): The name of the saved model file (default is 'model.pkl').

    Returns:
        model: The loaded model.
    """
    output_dir = f"./src/experiments/{experiment_name}/{run_name}"
    model_path = os.path.join(output_dir, 'model.pkl')
    model = joblib.load(model_path)

    return model


def get_best_run(experiment_name, run_name, metric_key='accuracy'):
    """
    Get the best run with the highest accuracy where the run name is 'baseline'.

    Parameters:
        experiment_id (str): The MLflow experiment ID.
        metric_key (str): The key of the metric to rank runs by (default is 'accuracy').

    Returns:
        best_run (pd.Series): The best run with the highest accuracy.
    """
    # Query for runs with the name 'baseline'
    runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id],
                              filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=[f"metrics.{metric_key} DESC"])

    # Get the best run (the first row will have the highest accuracy)
    if not runs.empty:
        best_run = runs.iloc[0]
        return best_run
    else:
        print(f"No runs found with the name '{run_name}'.")
        return None


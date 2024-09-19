import os
from datetime import datetime

import pandas as pd

from src.data_registry import DataRegistry
from src.modeling.utils import train_test_splitter, load_registered_model, load_state_config, get_predictions
import xgboost as xgb

from src.utils import find_year_for_season, get_seasons_to_update


def inference_pipeline(data_features, experiment_name, run_name, metric, feature_store_group, feature_store_name, feature_store_start_season=2002):
    """
    Inference pipeline for the baseline model. This will refit the model for each week/season and save the predictions.

    Parameters:
        data_features (list): List of features for the model.
        experiment_name (str): The name of the experiment.
        run_name (str): The name of the run.
        feature_store_group (str): Feature store group.
        feature_store_name (str): Feature store name.
        feature_store_start_season (int): The season to start from (default: 2002).
    """

    meta_cols = ['team', 'season', 'week', data_features[0]] if experiment_name == 'team_point' else ['home_team', 'away_team', 'season', 'week', data_features[0]]

    # Load the registered model
    print(f'Loading registered model: {experiment_name}/{run_name}')
    registered_clf = load_registered_model(experiment_name=experiment_name, run_name=run_name)
    if registered_clf is None:
        raise Exception(f"No Model Registered for {experiment_name}-{run_name} cant run inference pipeline")
    state_config = load_state_config(f"./src/experiments/{experiment_name}/{run_name}")
    model_uuid = state_config['run_id']
    hyperparameters = state_config['hyperparameters']


    #### Handle XGBoost vs Random Forest ####
    is_sklearn = not 'xgboost' in str(registered_clf.__class__)

    # Initialize data registry to load datasets
    data_registry = DataRegistry(feature_store_group=feature_store_group, feature_store_name=feature_store_name)
    df = data_registry.make_dataset(start_season=feature_store_start_season, end_season=find_year_for_season())
    current_week = df[((df['season'] == df.season.max()) & (df.away_score.notnull()))].week.max()
    df = df.fillna(0)

    # Get seasons to update
    root_path = f'./data/predictions/{experiment_name}/{run_name}'
    seasons_to_update = get_seasons_to_update(root_path=root_path)

    for season in seasons_to_update:
        new_season_predictions_df = pd.DataFrame()

        # Load previous season predictions if available
        fs_season_predictions_df = get_predictions(experiment_name=experiment_name, run_name=run_name, season=season)
        if fs_season_predictions_df is None:
            fs_season_predictions_df = pd.DataFrame()

        # Determine latest week based on available predictions
        if fs_season_predictions_df.empty:
            latest_week = 1
        else:
            latest_week = fs_season_predictions_df['week'].max() - 1

        print(f"Loaded Predictions from {season} - {latest_week - 1}...")

        # Set the max week (18 for a complete season, or less for the current season)
        max_week = 18 if season < find_year_for_season() else current_week + 1

        weeks = list(range(latest_week, max_week + 1))

        for week in weeks:
            print(f"Generating Predictions for {season} - {week}...")
            # Split data into training and testing
            X_train, y_train, X_test, y_test, metadata_test = train_test_splitter(
                df, data_features,
                start_season=feature_store_start_season,
                run_season=season, run_week=week,
                include_extra_week=season == find_year_for_season(),
                metadata_cols=meta_cols
            )

            # Re-fit the model but keep state
            weekly_clf = registered_clf
            if is_sklearn:
                weekly_clf.fit(X_train, y_train)
                y_pred = weekly_clf.predict_proba(X_test)[:, 1] if metric == 'accuracy' else weekly_clf.predict(X_test)
            else:
                hyperparams = hyperparameters
                hyperparams['early_stopping_rounds'] = None # No validation
                hyperparams['random_state'] = 0 # Making Sure State is same
                if metric == 'accuracy':
                    hyperparams['objective'] = 'reg:squarederror'
                    hyperparams['eval_metric'] = 'mae'
                    #weekly_clf = xgb.XGBClassifier(**hyperparams)
                else:
                    hyperparams['objective'] = 'binary:logistic'
                    hyperparams['eval_metric'] = 'error'
                    #weekly_clf = xgb.XGBRegressor(**hyperparams)
                dtrain = xgb.DMatrix(X_train, label=y_train)
                weekly_clf = xgb.train(hyperparams, dtrain, num_boost_round=1000) # Reduce num boost rounds for inference
                y_pred = weekly_clf.predict(xgb.DMatrix(X_test))

            # Predict probabilities
            metadata_test['prediction'] = y_pred
            metadata_test['model_uuid'] = model_uuid
            metadata_test['updated_at'] = datetime.now()

            # Collect predictions
            new_season_predictions_df = pd.concat([new_season_predictions_df, metadata_test])

        # Combine new predictions with previous ones (drop duplicates)
        combined_predictions = pd.concat([fs_season_predictions_df, new_season_predictions_df]).drop_duplicates(
            subset=['home_team', 'away_team', 'season', 'week'], keep='last'
        )

        # Save updated predictions
        os.makedirs(root_path, exist_ok=True)
        output_path = f'{root_path}/{season}.parquet'
        combined_predictions.to_parquet(output_path)
        print(f"Saved predictions for season {season} to {output_path}")
import os
from datetime import datetime

import pandas as pd

from src.data_registry import DataRegistry
from src.modeling.utils import get_predictions
from src.utils import find_year_for_season, get_seasons_to_update


def make_report(df, season, mode='inference'):
    upcoming_models = [
        {'experiment_name': 'team_point', 'run_name': 'baseline'},
        {'experiment_name': 'team_point', 'run_name': 'xgboost_optuna_tuner'},
    ]
    active_models = [
        {'experiment_name': 'win_percentage', 'run_name': 'baseline'},
        {'experiment_name': 'win_percentage', 'run_name': 'xgboost_optuna_tuner'},
    ]
    targets = [
        'home_score',
        'away_score',
        'away_team_win',
        'away_team_spread',
        'total_target',
        'away_team_covered',
        'home_team_covered',
        'under_covered',
        'away_team_covered_spread',
    ]
    report_df = df[[
                       'season',
                       'week',
                       'away_team',
                       'away_elo_pre',
                       'home_team',
                       'home_elo_pre',
                       'spread_line',
                       'total_line',
                       'away_elo_prob',
                       'home_elo_prob',
                   ] + targets].copy().rename(columns={
        'away_elo_pre': 'away_rating',
        'home_elo_pre': 'home_rating',
        'away_elo_prob': 'away_wp_elo',
        'home_elo_prob': 'home_wp_elo',
    })

    report_df['away_wp_vegas'] = report_df['spread_line'].apply(lambda x: 1 if x < 0 else 0)
    report_df[f"wp_elo_error_rate"] = (report_df[f"away_wp_elo"] - report_df.away_team_win).abs()
    report_df[f"wp_elo_result"] = ((report_df[f"away_wp_elo"] > 0.5) == report_df.away_team_win).astype(int)
    report_df[f"wp_vegas_result"] = ((report_df[f"away_wp_vegas"] > 0.5) == report_df.away_team_win).astype(int)

    for active_model in active_models:
        experiment_name = active_model['experiment_name']
        run_name = active_model['run_name']
        print(f"Adding {experiment_name} {run_name} to the inference report...")
        fs_season_predictions_df = get_predictions(experiment_name=experiment_name, run_name=run_name, season=season)
        if experiment_name == 'win_percentage':
            fs_season_predictions_df[f"away_wp_{run_name}"] = fs_season_predictions_df.prediction
            fs_season_predictions_df[f"home_wp_{run_name}"] = 1 - fs_season_predictions_df.prediction
            fs_season_predictions_df[f"wp_{run_name}_error_rate"] = (fs_season_predictions_df[f"away_wp_{run_name}"] - fs_season_predictions_df.away_team_win).abs()
            fs_season_predictions_df[f"wp_{run_name}_result"] = (((fs_season_predictions_df[f"away_wp_{run_name}"] > 0.5) == fs_season_predictions_df.away_team_win)).astype(int)

            report_df = pd.merge(report_df, fs_season_predictions_df[['season', 'week', 'away_team', 'home_team', f"away_wp_{run_name}", f"home_wp_{run_name}", f"wp_{run_name}_error_rate", f"wp_{run_name}_result"]], on=['season', 'week', 'away_team', 'home_team'], how='left')
        elif experiment_name == 'team_point':
            pass  # TODO
            # Team point should be broken down into spread (home_pred_score - away_pred_score) and total (home_pred_score + away_pred_score)

        elif experiment_name == 'spread':
            pass  # TODO
        else:
            pass  # TODO
    report_df['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if mode == 'inference':
        report_df = report_df.drop(columns=targets + [col for col in report_df.columns if '_error_rate' in col or '_result' in col])
    elif mode == 'evaluation':
        pass
        # TODO Create an evaluation report where instead of having the predictions as columns we have the result for each system
        # (Error rate for win_percentage, team_point, spread and total)
        # (1 for spread and total if the model correctly covered the spread_line or total_line and 0 otherwise)
        # (1 for win_percentage if the model correctly predicted the win_percentage and 0 otherwise)

    else:
        raise Exception("Invalid report mode: ", mode)
    return report_df

def get_report(season, type='inference'):
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
        file_path = f'./data/{type}_report/{season}.parquet'

        # Check if the file exists
        if os.path.exists(file_path):
            # Load the predictions from the parquet file
            predictions_df = pd.read_parquet(file_path)
            return predictions_df
        else:
            print(f"No report found for season {season}.")
            return None
    except Exception as e:
        print(f"Error occurred while retrieving report: {e}")
        return None

def weekly_report_builder():
    root_path = './data/weekly_inference_report'

    fs_season_report_df = get_report(season=find_year_for_season(), type='inference')
    for (week), sub_df in fs_season_report_df.groupby(['week']):
        week = week[0]
        sub_df['week'] = week
        os.makedirs(f"{root_path}/{week}", exist_ok=True)
        sub_df.to_csv(f"{root_path}/{week}/inference_report_{week}.csv", index=False)


def report_pipeline(feature_store_group, feature_store_name, feature_store_start_season=2002):
    """
    report pipeline for the models.

    Parameters:

    """

    # Initialize data registry to load datasets
    data_registry = DataRegistry(feature_store_group=feature_store_group, feature_store_name=feature_store_name)
    df = data_registry.make_dataset(start_season=feature_store_start_season, end_season=find_year_for_season())
    df = df.fillna(0)
    report_types = ['inference', 'evaluation']
    # Get seasons to update
    for report_type in report_types:
        root_path = f'./data/{report_type}_report'
        seasons_to_update = get_seasons_to_update(root_path=root_path)

        for season in seasons_to_update:
            new_season_report_df = make_report(df[df.season==season].copy(), season, mode=report_type)
            # Save updated report
            os.makedirs(root_path, exist_ok=True)
            output_path = f'{root_path}/{season}.parquet'
            new_season_report_df.to_parquet(output_path)
            print(f"Saved {report_type} report for season {season} to {output_path}")
    weekly_report_builder()
from src.experiments.win_percentage.win_percentage import win_percentage_baseline_inference_pipeline, win_percentage_xgb_optuna_tuner_inference_pipeline
from src.modeling.inference.report import report_pipeline

if __name__ == '__main__':
    FEATURE_STORE_GROUP = "event"
    FEATURE_STORE_NAME = "regular_season_game"
    ##################################################
    ### Win Percentage Models
    ##################################################
    win_percentage_baseline_inference_pipeline()
    win_percentage_xgb_optuna_tuner_inference_pipeline()

    report_pipeline(FEATURE_STORE_GROUP, FEATURE_STORE_NAME, 2022)
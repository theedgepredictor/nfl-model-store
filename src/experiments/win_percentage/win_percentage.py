from src.modeling.inference.pipeline import inference_pipeline
from src.utils import get_current_season, get_current_week

EXPERIMENT_NAME = "win_percentage"
FEATURE_STORE_GROUP = "event"
FEATURE_STORE_NAME = "regular_season_game"
METRIC = 'accuracy'


def win_percentage_baseline_inference_pipeline():
    baseline_features = [
        'away_team_win',
        # 'season',
        # 'week',
        # 'home_team',
        # 'away_team',
        # 'home_score',
        # 'away_score',
        'spread_line',
        'total_line',
        # 'away_team_spread',
        # 'total_target',
        # 'away_team_covered',
        # 'home_team_covered',
        # 'under_covered',
        # 'away_team_covered_spread',
        'away_ewma_rushing_offense',
        'away_ewma_rushing_defense',
        'away_ewma_passing_offense',
        'away_ewma_passing_defense',
        'away_ewma_score_offense',
        'away_ewma_score_defense',
        'home_ewma_rushing_offense',
        'home_ewma_rushing_defense',
        'home_ewma_passing_offense',
        'home_ewma_passing_defense',
        'home_ewma_score_offense',
        'home_ewma_score_defense',
        # 'away_rolling_spread_cover',
        # 'away_rolling_under_cover',
        # 'home_rolling_spread_cover',
        # 'home_rolling_under_cover',
        'away_elo_pre',
        # 'away_elo_prob',
        'home_elo_pre',
        # 'home_elo_prob'
    ]
    inference_pipeline(
        data_features=baseline_features,
        experiment_name=EXPERIMENT_NAME,
        run_name='baseline',
        metric=METRIC,
        feature_store_group=FEATURE_STORE_GROUP,
        feature_store_name=FEATURE_STORE_NAME,
        feature_store_start_season=2002
    )

def win_percentage_xgb_optuna_tuner_inference_pipeline():
    features = [
        'away_team_win',
        'spread_line',
        'away_ewma_rushing_offense',
        'away_ewma_rushing_defense',
        'away_ewma_passing_offense',
        'away_ewma_passing_defense',
        'away_ewma_score_offense',
        'away_ewma_score_defense',
        'home_ewma_rushing_offense',
        'home_ewma_rushing_defense',
        'home_ewma_passing_offense',
        'home_ewma_passing_defense',
        'home_ewma_score_offense',
        'home_ewma_score_defense',
        'away_elo_pre',
        'home_elo_pre',
        'home_avg_points_offense',
        'home_avg_points_defense',
        'home_avg_point_differential_offense',
        'home_avg_point_differential_defense',
        'home_avg_fantasy_points_ppr_offense',
        'home_avg_fantasy_points_ppr_defense',
        'home_avg_total_yards_offense',
        'home_avg_total_yards_defense',
        'home_avg_total_turnovers_offense',
        'home_avg_total_turnovers_defense',
        'home_avg_total_touchdowns_offense',
        'home_avg_total_touchdowns_defense',
        'home_avg_yards_per_play_offense',
        'home_avg_yards_per_play_defense',
        'home_avg_qbr_offense',
        'home_avg_qbr_defense',
        'home_avg_yards_per_pass_attempt_offense',
        'home_avg_yards_per_pass_attempt_defense',
        'home_avg_third_down_percentage_offense',
        'home_avg_third_down_percentage_defense',
        'away_avg_points_offense',
        'away_avg_points_defense',
        'away_avg_point_differential_offense',
        'away_avg_point_differential_defense',
        'away_avg_fantasy_points_ppr_offense',
        'away_avg_fantasy_points_ppr_defense',
        'away_avg_total_yards_offense',
        'away_avg_total_yards_defense',
        'away_avg_total_turnovers_offense',
        'away_avg_total_turnovers_defense',
        'away_avg_total_touchdowns_offense',
        'away_avg_total_touchdowns_defense',
        'away_avg_yards_per_play_offense',
        'away_avg_yards_per_play_defense',
        'away_avg_qbr_offense',
        'away_avg_qbr_defense',
        'away_avg_yards_per_pass_attempt_offense',
        'away_avg_yards_per_pass_attempt_defense',
        'away_avg_third_down_percentage_offense',
        'away_avg_third_down_percentage_defense'
    ]
    inference_pipeline(
        data_features=features,
        experiment_name=EXPERIMENT_NAME,
        run_name='xgboost_optuna_tuner',
        metric=METRIC,
        feature_store_group=FEATURE_STORE_GROUP,
        feature_store_name=FEATURE_STORE_NAME,
        feature_store_start_season=2002
    )


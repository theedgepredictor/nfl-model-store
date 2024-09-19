from datetime import datetime
import mlflow
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from src.modeling.tuning.xgboost_optuna_tuner import XgboostOptunaTuner
from src.modeling.utils import register_model, save_state_config, train_test_splitter, load_state_config, get_best_run
import xgboost as xgb

def mlflow_trainer(
        df,
        data_columns,
        experiment_name,
        run_name,
        metric,
        metric_threshold,
        run_season,
        run_week,
        feature_store_group,
        feature_store_name,
        model_class,
        start_season,
        n_trials=75
):
    """

    """
    meta_cols = ['team', 'season', 'week', data_columns[0]] if experiment_name == 'team_point' else ['home_team', 'away_team', 'season', 'week', data_columns[0]]

    state_config = {
        'data_columns': data_columns,
        'experiment_name': experiment_name,
        'run_name': run_name,
        'hyperparameters': None,
        'experiment_id': None,
        'run_id': None,
        'previous_registered_run_id': None,
        'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name)

    mlflow.set_experiment(experiment_name=experiment_name)

    best_run = get_best_run(experiment_name=experiment_name, run_name=run_name)
    if best_run is not None and hasattr(best_run, f'metrics.{metric}'):
        best_metric = best_run[f'metrics.{metric}']
        prev_state_config = load_state_config(root_path=f"./src/experiments/{experiment_name}/{run_name}")
        state_config['previous_registered_run_id'] = best_run['run_id'] if prev_state_config is None else prev_state_config['run_id']
        print(f"Best run to beat: {best_run['run_id']} with {metric} = {best_metric}")
    else:
        print(f"No Successful runs found with the name '{run_name}' using default threshold of {metric_threshold}.")
        best_metric = metric_threshold

    with mlflow.start_run(run_name=run_name) as run:
        print('MLFlow Run:')
        print(f'    experiment_id: {run.info.experiment_id}')
        print(f'    run_id: {run.info.run_id}')
        print(f'    experiment_name: {experiment_name}')
        print(f'    run_name: {run_name}')
        state_config['run_id'] = run.info.run_id
        state_config['experiment_id'] = run.info.experiment_id

        X_train, y_train, X_test, y_test, metadata_test = train_test_splitter(df, data_columns, start_season=start_season, run_season=run_season, metadata_cols=meta_cols)

        mlflow.set_tags(
            {
                'data_features': data_columns,
                'feature_store_group': feature_store_group,
                'feature_store_name': feature_store_name,
                'model_name': model_class,
                'run_season': run_season,
                'run_week': run_week,
                'start_season': start_season,
                'task': 'Classification' if metric == 'accuracy' else 'Regression',
                'task_primary_metric': metric,
                'task_primary_metric_threshold': metric_threshold,
            }
        )
        if run_name == 'xgboost_optuna_tuner':
            tuner = XgboostOptunaTuner(X_train, y_train, X_test, y_test, task='classification' if metric == 'accuracy' else 'regression', n_trials=n_trials)
            best_params = tuner.run_optimization()
            clf = tuner.fit_predict(best_params)
            y_pred = clf.predict(xgb.DMatrix(X_test))
        else:
            best_params = {
                'n_estimators': 500,
                # 'class_weight': 'balanced',
                # 'criterion': 'log_loss',
                # 'max_depth': 15,
                'random_state': 0
            }
            clf = RandomForestClassifier(**best_params) if metric == 'accuracy' else RandomForestRegressor(**best_params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1] if metric == 'accuracy' else clf.predict(X_test)
        mlflow.log_params(best_params)
        state_config['hyperparameters'] = best_params

        val_metric = accuracy_score(y_test, y_pred > 0.5) if metric == 'accuracy' else mean_absolute_error(y_test, y_pred)
        mlflow.log_metric(key=metric, value=val_metric)

        if val_metric > metric_threshold:
            if run_name == 'xgboost_optuna_tuner':
                mlflow.xgboost.log_model(clf, 'model')
            else:
                mlflow.sklearn.log_model(clf, 'model')
        else:
            print(f"Model {metric}: {val_metric} < {metric_threshold}. Skipping model export.")

        if val_metric >= best_metric:
            print('NEW BEST MODEL! Overwriting previous best model...')

            saved_model_path = register_model(clf, experiment_name=experiment_name, run_name=run_name)
            save_state_config(state_config, root_path=f"./src/experiments/{experiment_name}/{run_name}")
            print(f"Model registered at: {saved_model_path}")
        mlflow.end_run()
###########################################################
## Modeling
###########################################################

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import KFold


def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000
    x = preds - labels
    grad = x / (x ** 2 / c ** 2 + 1)
    hess = -c ** 2 * (x ** 2 - c ** 2) / (x ** 2 + c ** 2) ** 2
    return grad, hess


class XgboostOptunaTuner:
    def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            X_val: pd.DataFrame,
            y_val: pd.DataFrame,
            task: str = 'regression',  # 'regression' or 'classification'
            n_trials: int = 50,
            custom_obj = None
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.task = task
        self.n_trials = n_trials
        self.hyperparameters = None
        self.model = None
        self.performance = None
        self.custom_obj = custom_obj
        self.trials = []
        self.study = None

    def objective(self, trial: optuna.Trial):
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "error",
            # use exact for small dataset.
            #"tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 2, 12)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-error")
        if self.task == 'regression':
            param['objective'] = 'reg:squarederror'
            param['eval_metric'] = 'mae'
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-mae")

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)

        history = xgb.cv(
            param,
            dtrain,
            obj=self.custom_obj if self.custom_obj is not None else None,
            nfold=5,
            early_stopping_rounds=50,
            verbose_eval=500,
            num_boost_round=4000,
            callbacks=[pruning_callback]
        )
        if self.task == 'regression':
            return history["test-mae-mean"].values[-1]
        else:
            return history["test-error-mean"].values[-1]

    def run_optimization(self):
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
        study = optuna.create_study(pruner=pruner, direction='minimize')
        study.optimize(lambda trial: self.objective(trial), n_trials=self.n_trials, n_jobs=-1, show_progress_bar=True)
        self.trials = study.trials
        self.study = study
        self.hyperparameters = study.best_params
        return self.hyperparameters

    def fit_predict(self, hyperparameters):
        self.hyperparameters = hyperparameters

        if self.task == 'regression':
            self.hyperparameters['objective'] = 'reg:squarederror'
            self.hyperparameters['eval_metric'] = 'mae'
        else:
            self.hyperparameters['objective'] = 'binary:logistic'
            self.hyperparameters['eval_metric'] = 'error'

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        dtest = xgb.DMatrix(self.X_val, label=self.y_val)

        for train_index, test_index in kfold.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
            y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[test_index]

            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
            bst = xgb.train(
                self.hyperparameters,
                dtrain,
                obj=self.custom_obj,
                num_boost_round=3000,
                evals=[(dval, 'validation')],
                early_stopping_rounds=50,
            )
            y_pred_test = bst.predict(dtest)

            if self.task == 'regression':
                score = mean_absolute_error(self.y_val, y_pred_test)
            else:
                score = accuracy_score(self.y_val, (y_pred_test > 0.5).astype(int))
            scores.append(score)

        if self.task == 'regression':
            print(f'Test MAE: {np.mean(scores):.4f}')
            self.performance = {'metric': 'mae', 'avg': np.mean(scores), 'min': np.min(scores), 'max': np.max(scores)}
        else:
            print(f'Test Accuracy: {np.mean(scores):.4f}')
            self.performance = {'metric': 'accuracy', 'avg': np.mean(scores), 'min': np.min(scores), 'max': np.max(scores)}

        self.model = self.fit_final(self.hyperparameters)
        return self.model

    def fit_final(self, hyperparameters):
        self.hyperparameters = hyperparameters

        if self.task == 'regression':
            self.hyperparameters['objective'] = 'reg:squarederror'
            self.hyperparameters['eval_metric'] = 'mae'
        else:
            self.hyperparameters['objective'] = 'binary:logistic'
            self.hyperparameters['eval_metric'] = 'error'

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        self.model = xgb.train(self.hyperparameters, dtrain, num_boost_round=3000, evals=[(dval, 'validation')], early_stopping_rounds=50)
        return self.model

    """
    def fit_final(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.hyperparameters['n_estimators'] = 3000
        self.hyperparameters['early_stopping_rounds'] = 50
        self.hyperparameters['random_state'] = 0
        if self.task == 'regression':
            self.hyperparameters['objective'] = 'reg:squarederror'
            self.hyperparameters['eval_metric'] = 'mae'
            self.model = xgb.XGBRegressor(**hyperparameters)
        else:
            self.hyperparameters['objective'] = 'binary:logistic'
            self.hyperparameters['eval_metric'] = 'error'
            self.model = xgb.XGBClassifier(**hyperparameters)

        #dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        #dval = xgb.DMatrix(self.X_val, label=self.y_val)
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=True
        )

        return self.model
    """
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
import optuna
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureSelection:
    """
    return a list of features
    """
    def __init__(self, X, y, n_features=25, visualize=False):
        self.X = X
        self.y = y
        self.n_features = n_features
        self.features = {}
        self.selected_features = None
        self.visualize = visualize

    def univariate_feature_selection(self):
        """
        When to Use:
        - Initial Screening: When you have a large number of features and you want to perform an initial screening to identify the most relevant features.
        - Simple and Fast: Useful for quick, simple problems where you need to identify strong individual relationships between each feature and the target variable.

        Advantages:
        - Simplicity: Easy to understand and implement.
        - Speed: Computationally efficient, especially for large datasets.

        Disadvantages:
        - Ignores Feature Interactions: Only considers each feature independently, without considering interactions between features.
        """
        if not self.features:
            self.features = {}
        self.features['univariate'] = self._univariate_feature_selection()
        if self.visualize:
            self.visualize_features('univariate')
        return self.features['univariate']

    def rfe_feature_selection(self):
        """
        When to Use:
        - Feature Interactions: When you suspect that interactions between features are important and you want a method that takes these into account.
        - Model-Based Selection: When you want to incorporate model performance into the feature selection process.

        Advantages:
        - Considers Interactions: Takes into account the contribution of each feature in the context of other features.
        - Model-Based: Uses the performance of a machine learning model to guide feature selection.

        Disadvantages:
        - Computationally Intensive: Can be slow, especially for large datasets and complex models.
        - Model-Specific: The selected features may be biased towards the specific model used in the RFE process.
        """

        if not self.features:
            self.features = {}
        self.features['rfe'] = self.rfe_feature_selection()
        if self.visualize:
            self.visualize_features('rfe')

    def random_forest_feature_importance(self):
        """
        When to Use:
        - Non-Linear Relationships: When you want to capture non-linear relationships between features and the target variable.
        - Model-Based Selection: When you prefer an ensemble method that can handle a large number of features and provide importance scores.

        Advantages:
        - Handles Non-Linearities: Effective for capturing complex relationships.
        - Robust to Overfitting: Random forests are generally robust to overfitting, especially with a large number of features.

        Disadvantages:
        - Bias Towards Continuous Features: Can be biased towards features with many categories or continuous variables.
        - Computationally Intensive: Training random forests can be computationally expensive.
        """
        if not self.features:
            self.features = {}
        self.features['random_forest'] = self._random_forest_feature_importance()
        if self.visualize:
            self.visualize_features('random_forest')

    def lasso_feature_selection(self):
        """
        When to Use:
        - High-Dimensional Data: Particularly useful when the number of features exceeds the number of observations.
        - Regularization Needs: When you need a method that performs both feature selection and regularization to prevent overfitting.

        Advantages:
        - Feature Selection and Regularization: Simultaneously performs feature selection and regularization, reducing model complexity.
        - Handles Multicollinearity: Can handle multicollinearity among features by selecting one feature from a group of highly correlated features.

        Disadvantages:
        - Linear Relationships: Assumes linear relationships between features and the target variable.
        - Bias: Can introduce bias in the model due to the shrinkage of coefficients.
        """
        if not self.features:
            self.features = {}
        self.features['lasso'] = self._lasso_feature_selection()
        if self.visualize:
            self.visualize_features('lasso')

    def correlation_analysis(self):
        """
        When to Use:
        - Initial Data Exploration: Useful during the initial exploration of the dataset to identify highly correlated features.
        - Multicollinearity Reduction: When you want to reduce multicollinearity by removing redundant features.

        Advantages:
        - Simple to Implement: Easy to calculate and interpret.
        - Reduces Multicollinearity: Helps in identifying and removing multicollinear features.

        Disadvantages:
        - Ignores Non-Linear Relationships: Only captures linear relationships between features.
        - Ignores Interaction Effects: Does not consider interactions between features.
        """
        if not self.features:
            self.features = {}
        self.features['correlation'] = self._correlation_analysis()
        if self.visualize:
            self.visualize_features('correlation')

    def optuna_feature_selection(self, n_trials=50):
        """
        When to Use:
        - Hyperparameter Optimization: When you want to combine feature selection with hyperparameter optimization for a specific model.
        - Complex Models: When dealing with complex models where traditional feature selection methods may not perform well.

        Advantages:
        - Combines Optimization: Simultaneously optimizes model hyperparameters and feature selection.
        - Flexibility: Can be used with any model and objective function.

        Disadvantages:
        - Computationally Intensive: Can be computationally expensive and time-consuming, especially for large datasets and many trials.
        - Requires Expertise: Requires a good understanding of the optimization process and hyperparameter tuning.
        """
        if not self.features:
            self.features = {}
        self.features['optuna'] = self._optuna_feature_selection(n_trials=n_trials)
        if self.visualize:
            self.visualize_features('optuna')

    def kitchen_sink_feature_selection(self):
        self.features = self._get_all_features()
        all_features = []
        for method, features in self.features.items():
            all_features.extend([feature["name"] for feature in features])

        feature_counter = Counter(all_features)
        common_features = [feature for feature, count in feature_counter.most_common(self.n_features)]

        selected_feature_importances = []
        for feature_name in common_features:
            for method, features in self.features.items():
                for feature in features:
                    if feature["name"] == feature_name:
                        selected_feature_importances.append(feature)
                        break

        self.selected_features = selected_feature_importances

        return self.selected_features

    def pipeline_feature_selection(self, method='lasso'):
        """
        Automated pipeline for performing:
        1. Correlation analysis
        2. Univariate Feature Selection (50% of all features)
        3. Step 3 Method Selection (lasso, rfe, random_forest, or optuna)
        """

        initial_features = self.X.columns

        # Step 1: Correlation Analysis
        corr_features = self._correlation_analysis()
        corr_features_names = [feature['name'] for feature in corr_features]
        self.X = self.X[corr_features_names]

        # Step 2: Univariate Feature Selection (50% of all features)
        temp_features = self.n_features
        n_features_univariate = max(1, len(corr_features_names) // 2)
        self.n_features = n_features_univariate
        univariate_features = self._univariate_feature_selection()
        univariate_features_names = [feature['name'] for feature in univariate_features]
        self.X = self.X[univariate_features_names]
        self.n_features = temp_features

        # Step 3: Method Selection
        step3_methods = {
            'lasso': self._lasso_feature_selection,
            'rfe': self._rfe_feature_selection,
            'random_forest': self._random_forest_feature_importance,
            'optuna': self._optuna_feature_selection
        }

        if method not in step3_methods:
            raise ValueError(f"Invalid method: {method}. Choose from 'lasso', 'rfe', 'random_forest', or 'optuna'.")

        self.features[method] = step3_methods[method]()
        step3_features_names = [feature['name'] for feature in self.features[method]]
        self.X = self.X[step3_features_names]

        # Features Dropped
        dropped_features = list(set(initial_features) - set(self.X.columns))
        print(f"Dropped features: {dropped_features}")
        print(f"Resulting Dimensions: {self.X.shape}")
        if self.visualize:
            self.visualize_features(method)
        return self.X.columns

    def visualize_features(self, method):
        """
        Visualize the results of a single method with matplotlib
        """
        features = self.features.get(method, [])
        if not features:
            print(f"No features found for method: {method}")
            return

        feature_names = [feature['name'] for feature in features]
        feature_values = [feature['value'] for feature in features]

        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_values, y=feature_names, palette='viridis', hue=feature_names, legend=False)
        plt.title(f'Feature Importances: {method}')
        plt.xlabel('Importance Value')
        plt.ylabel('Feature Name')
        plt.show()

    def get_selected_features(self, method):
        return self.features.get(method, [])

    def get_selected_features_names(self, method):
        return [feature['name'] for feature in self.features.get(method, [])]

    def _univariate_feature_selection(self):
        selector = SelectKBest(score_func=f_regression, k=self.n_features)
        selector.fit(self.X, self.y)
        scores = selector.scores_[selector.get_support()]
        feature_names = self.X.columns[selector.get_support()]
        return [{"name": name, "value": score, "rank": rank + 1} for rank, (name, score) in enumerate(sorted(zip(feature_names, scores), key=lambda x: -x[1]))]

    def _rfe_feature_selection(self):
        model = RandomForestRegressor(random_state=42)
        rfe = RFE(model, n_features_to_select=self.n_features)
        rfe.fit(self.X, self.y)
        ranking = rfe.ranking_[rfe.support_]
        feature_names = self.X.columns[rfe.get_support()]
        return [{"name": name, "value": 1 / rank, "rank": rank} for name, rank in zip(feature_names, ranking)]

    def _random_forest_feature_importance(self):
        model = RandomForestRegressor(random_state=42)
        model.fit(self.X, self.y)
        importances = model.feature_importances_
        feature_importance = pd.Series(importances, index=self.X.columns).sort_values(ascending=False)
        return [{"name": name, "value": value, "rank": rank + 1} for rank, (name, value) in enumerate(feature_importance.head(self.n_features).items())]

    def _lasso_feature_selection(self):
        lasso = LassoCV(random_state=42)
        lasso.fit(self.X, self.y)
        coef = lasso.coef_[lasso.coef_ != 0]
        feature_names = self.X.columns[lasso.coef_ != 0]
        return [{"name": name, "value": abs(value), "rank": rank + 1} for rank, (name, value) in enumerate(sorted(zip(feature_names, coef), key=lambda x: -abs(x[1])))]

    def _correlation_analysis(self):
        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        remaining_features = self.X.columns.difference(to_drop).tolist()[:self.n_features]
        print(f"Dropped {len(to_drop)} columns due to high correlation: {to_drop}")
        return [{"name": name, "value": 1, "rank": rank + 1} for rank, name in enumerate(remaining_features)]

    def _optuna_feature_selection(self, n_trials=50):
        def objective(trial):
            n_features = trial.suggest_int('n_features', 5, self.n_features)
            selector = SelectKBest(score_func=f_regression, k=n_features)
            X_new = selector.fit_transform(self.X, self.y)
            score = cross_val_score(RandomForestRegressor(random_state=42), X_new, self.y, cv=5, scoring='neg_mean_squared_error').mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_n_features = study.best_params['n_features']
        selector = SelectKBest(score_func=f_regression, k=best_n_features)
        selector.fit(self.X, self.y)
        scores = selector.scores_[selector.get_support()]
        feature_names = self.X.columns[selector.get_support()]
        return [{"name": name, "value": score, "rank": rank + 1} for rank, (name, score) in enumerate(sorted(zip(feature_names, scores), key=lambda x: -x[1]))]

    def _get_all_features(self):
        methods = {
            "univariate": self._univariate_feature_selection,
            "rfe": self._rfe_feature_selection,
            "random_forest": self._random_forest_feature_importance,
            "lasso": self._lasso_feature_selection,
            "correlation": self._correlation_analysis,
            "optuna": self._optuna_feature_selection
        }
        return {method: method_func() for method, method_func in methods.items()}
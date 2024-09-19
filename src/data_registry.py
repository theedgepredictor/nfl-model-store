## Add feature store connection, ML tracking object and dataset CRUD operations
from typing import List

import pandas as pd

from src.utils import get_dataframe


class DataRegistry:
    def __init__(self, feature_store_group: str, feature_store_name: str):
        self.feature_store_group = feature_store_group
        self.feature_store_name = feature_store_name
        self.feature_store_url = f"https://github.com/theedgepredictor/nfl-feature-store/raw/main/data/feature_store/{self.feature_store_group}/{self.feature_store_name}"

    def get_season(self, season: int):
        return get_dataframe(f"{self.feature_store_url}/{season}.parquet")

    def make_dataset(self, start_season: int, end_season: int):
        return pd.concat([self.get_season(season) for season in list(range(start_season, end_season+1))], ignore_index=True)
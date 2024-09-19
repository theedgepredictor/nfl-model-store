import joblib
import mlflow
import numpy as np
import pandas as pd
import datetime
import os
from typing import List
import pyarrow as pa
from espn_api_orm.calendar.api import ESPNCalendarAPI
from espn_api_orm.consts import ESPNSportTypes
from espn_api_orm.league.api import ESPNLeagueAPI
from pandas.core.dtypes.common import is_numeric_dtype


def get_dataframe(path: str, columns: List = None):
    """
    Read a DataFrame from a parquet file.

    Args:
        path (str): Path to the parquet file.
        columns (List): List of columns to select (default is None).

    Returns:
        pd.DataFrame: Read DataFrame.
    """
    try:
        return pd.read_parquet(path, engine='pyarrow', dtype_backend='numpy_nullable', columns=columns)
    except Exception as e:
        print(e)
        return pd.DataFrame()


def put_dataframe(df: pd.DataFrame, path: str):
    """
    Write a DataFrame to a parquet file.

    Args:
        df (pd.DataFrame): DataFrame to write.
        path (str): Path to the parquet file.
        schema (dict): Schema dictionary.

    Returns:
        None
    """
    key, file_name = path.rsplit('/', 1)
    if file_name.split('.')[1] != 'parquet':
        raise Exception("Invalid Filetype for Storage (Supported: 'parquet')")
    os.makedirs(key, exist_ok=True)
    df.to_parquet(f"{key}/{file_name}",engine='pyarrow', schema=pa.Schema.from_pandas(df))


def create_dataframe(obj, schema: dict):
    """
    Create a DataFrame from an object with a specified schema.

    Args:
        obj: Object to convert to a DataFrame.
        schema (dict): Schema dictionary.

    Returns:
        pd.DataFrame: Created DataFrame.
    """
    df = pd.DataFrame(obj)
    for column, dtype in schema.items():
        df[column] = df[column].astype(dtype)
    return df


def get_current_season():
    league_api = ESPNLeagueAPI(ESPNSportTypes.FOOTBALL, 'nfl')
    return league_api.get_current_season()

def get_current_week():
    season = get_current_season()
    calendar_api = ESPNCalendarAPI(ESPNSportTypes.FOOTBALL, 'nfl', season=season)
    for season_type in [2,3]:
        for week in calendar_api.get_weeks(season_type):
            res = calendar_api.api_request(calendar_api._core_url + f"/{calendar_api.sport.value}/leagues/{calendar_api.league}/seasons/{calendar_api.season}/types/{season_type}/weeks/{week}")
            if datetime.datetime.now() < datetime.datetime.strptime(res['endDate'], "%Y-%m-%dT%H:%MZ"):
                return week + 18 if season_type == 3 else week

def df_rename_pivot(df, all_cols, pivot_cols, t1_prefix, t2_prefix, sub_merge_df=None):
    '''
    The reverse of a df_rename_fold
    Pivot one generic type into two prefixed column types
    Ex: team_id -> away_team_id and home_team_id
    '''
    try:
        df = df[all_cols]
        t1_cols = [t1_prefix + i for i in all_cols if i not in pivot_cols]
        t2_cols = [t2_prefix + i for i in all_cols if i not in pivot_cols]
        original_cols = [i for i in all_cols if i not in pivot_cols]

        t1_renamed_pivot_df = df.rename(columns=dict(zip(original_cols, t1_cols)))
        t2_renamed_pivot_df = df.rename(columns=dict(zip(original_cols, t2_cols)))

        if sub_merge_df is None:
            df_out = pd.merge(t1_renamed_pivot_df, t2_renamed_pivot_df, on=pivot_cols).reset_index().drop(columns='index')
        else:
            sub_merge_cols = sub_merge_df.columns.values
            t1_sub_df = pd.merge(sub_merge_df, t1_renamed_pivot_df, how='inner', left_on=[t1_prefix + i for i in pivot_cols], right_on=pivot_cols).drop(columns=pivot_cols)
            t2_sub_df = pd.merge(sub_merge_df, t2_renamed_pivot_df, how='inner', left_on=[t2_prefix + i for i in pivot_cols], right_on=pivot_cols).drop(columns=pivot_cols)
            df_out = pd.merge(t1_sub_df, t2_sub_df, on=list(sub_merge_cols))
        return df_out
    except Exception as e:
        print("--df_rename_pivot-- " + str(e))
        print(f"columns in: {df.columns}")
        print(f"shape: {df.shape}")
        return df


def df_rename_fold(df, t1_prefix, t2_prefix):
    '''
    The reverse of a df_rename_pivot
    Fold two prefixed column types into one generic type
    Ex: away_team_id and home_team_id -> team_id
    '''
    try:
        t1_all_cols = [i for i in df.columns if t2_prefix not in i]
        t2_all_cols = [i for i in df.columns if t1_prefix not in i]

        t1_cols = [i for i in df.columns if t1_prefix in i]
        t2_cols = [i for i in df.columns if t2_prefix in i]
        t1_new_cols = [i.replace(t1_prefix, '') for i in df.columns if t1_prefix in i]
        t2_new_cols = [i.replace(t2_prefix, '') for i in df.columns if t2_prefix in i]

        t1_df = df[t1_all_cols].rename(columns=dict(zip(t1_cols, t1_new_cols)))
        t2_df = df[t2_all_cols].rename(columns=dict(zip(t2_cols, t2_new_cols)))

        df_out = pd.concat([t1_df, t2_df]).reset_index().drop(columns='index')
        return df_out
    except Exception as e:
        print("--df_rename_fold-- " + str(e))
        print(f"columns in: {df.columns}")
        print(f"shape: {df.shape}")
        return df


def df_rename_dif(df, t1_prefix=None, t2_prefix=None, t1_cols=None, t2_cols=None, sub_prefix=''):
    '''
    An extension of the df_rename_pivot
    Take the difference of two prefixed column types
    Ex: away_team_turnovers - home_team_turnovers -> team_turnovers_dif
    Note: This method applies the difference to the columns and removes the two prefixed column types
    '''
    if t1_cols is None and t2_cols is None:
        if t1_prefix is None or t2_prefix is None:
            raise Exception('You must specify either prefix or cols')
        t1_cols = [i for i in df.columns if t1_prefix in i]
        t2_cols = [i for i in df.columns if t2_prefix in i]
    for t1_col, t2_col in zip(t1_cols, t2_cols):
        if is_numeric_dtype(df[t1_col]) and is_numeric_dtype(df[t2_col]):
            df[f"dif_{t1_col.replace(t1_prefix, sub_prefix)}"] = df[t1_col] - df[t2_col]
    df_out = df.drop(columns=t1_cols + t2_cols)
    return df_out

def df_rename_shift(df, drop_cols=None):
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)

    root_cols = [col for col in df.columns if 'off_' not in col and 'def_' not in col and 'away_' not in col and 'home_' not in col]

    away_cols = [col for col in df.columns if 'off_' not in col and 'def_' not in col and 'away_' in col and 'home_' not in col]
    away_rename_dict = {col: col.replace('away_', '') for col in away_cols}
    home_cols = [col for col in df.columns if 'off_' not in col and 'def_' not in col and 'away_' not in col and 'home_' in col]
    home_rename_dict = {col: col.replace('home_', '') for col in home_cols}

    off_away_cols = [col for col in df.columns if 'off_' in col and 'away_' in col]
    off_away_rename_dict = {col: col.replace('away_', '') for col in off_away_cols}
    def_away_cols = [col for col in df.columns if 'def_' in col and 'away_' in col]
    def_away_rename_dict = {col: col.replace('away_', '') for col in def_away_cols}

    off_home_cols = [col for col in df.columns if 'off_' in col and 'home_' in col]
    off_home_rename_dict = {col: col.replace('home_', '') for col in off_home_cols}
    def_home_cols = [col for col in df.columns if 'def_' in col and 'home_' in col]
    def_home_rename_dict = {col: col.replace('home_', '') for col in def_home_cols}

    away_df = df[root_cols + away_cols + off_away_cols + def_home_cols].rename(columns={**away_rename_dict, **off_away_rename_dict, **def_home_rename_dict})
    away_df['is_home'] = 0
    home_df = df[root_cols + home_cols + off_home_cols + def_away_cols].rename(columns={**home_rename_dict, **off_home_rename_dict, **def_away_rename_dict})
    home_df['is_home'] = 1
    del df
    out_df = pd.concat([away_df, home_df])
    return out_df


def get_seasons_to_update(root_path):
    """
    Get a list of seasons to update based on the root path and sport.

    Args:
        root_path (str): Root path for the sport data.
        sport (ESPNSportTypes): Type of sport.

    Returns:
        List: List of seasons to update.
    """
    current_season = find_year_for_season()
    if os.path.exists(f'{root_path}'):
        seasons = os.listdir(f'{root_path}')
        fs_season = -1
        for season in seasons:
            temp = int(season.split('.')[0])
            if temp > fs_season:
                fs_season = temp
    else:
        fs_season = 2022
    if fs_season == -1:
        fs_season = 2022
    return list(range(fs_season, current_season + 1))

def find_year_for_season( date: datetime.datetime = None):
    """
    Find the year for a specific season based on the league and date.

    Args:
        league (ESPNSportTypes): Type of sport.
        date (datetime.datetime): Date for the sport (default is None).

    Returns:
        int: Year for the season.
    """
    SEASON_START_MONTH = {

        "NFL": {'start': 8, 'wrap': False},
    }
    if date is None:
        today = datetime.datetime.utcnow()
    else:
        today = date
    start = SEASON_START_MONTH["NFL"]['start']
    wrap = SEASON_START_MONTH["NFL"]['wrap']
    if wrap and start - 1 <= today.month <= 12:
        return today.year + 1
    elif not wrap and start == 1 and today.month == 12:
        return today.year + 1
    elif not wrap and not start - 1 <= today.month <= 12:
        return today.year - 1
    else:
        return today.year
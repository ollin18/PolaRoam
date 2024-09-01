# %%
import os
# os.environ['POLARS_MAX_THREADS'] = '20'
import pandas as pd
import polars as pl
import polaroam
import models
from models import Infostop
import numpy as np
import utils
import time

# %%

df = pl.scan_parquet("~/data_quadrant/ollin_clusters/all_days_clusters.parquet")

df.collect_schema()

def pandas_labels_home(key,data):
        user_df = data
        user_df = initialize_user_df(user_df)
        home_tmp = get_home_tmp(user_df, start_hour_day, end_hour_day)
        if home_tmp.empty:
            return user_df
        home_tmp = compute_cumulative_duration(home_tmp, home_period_window, min_periods_over_window, min_pings_home_cluster_label)
        if home_tmp.empty:
            return user_df
        user_df = add_home_label(user_df, home_tmp)
        user_df = interpolate_missing_dates(user_df, home_tmp)
        return remove_unused_cols(user_df)#user_df) #if home_tmp.cluster_label.unique().size != 0 else user_df.drop(['location_type', 'home_label'], axis=1)

def initialize_user_df(user_df):
    user_df['location_type'] = 'O'
    user_df['home_label'] = -1
    user_df['work_label'] = -1
    return user_df

def get_home_tmp(user_df, start_hour_day, end_hour_day):
    return user_df[(user_df['t_start_hour'] >= end_hour_day) | (user_df['t_end_hour'] <= start_hour_day)].copy()

def compute_cumulative_duration(home_tmp, home_period_window, min_periods_over_window, min_pings_home_cluster_label):
    home_tmp = home_tmp[['cluster_label', 'date_trunc', 'duration', 'total_pings_stop']].groupby(['cluster_label', 'date_trunc']).sum().reset_index().sort_values('date_trunc')
    home_tmp = home_tmp.merge(home_tmp[['date_trunc', 'cluster_label', 'duration', 'total_pings_stop']].groupby(['cluster_label']).apply(home_rolling_on_date, home_period_window, min_periods_over_window).reset_index(), on=['date_trunc', 'cluster_label'], suffixes=('', '_cum'))
    home_tmp = home_tmp[home_tmp.total_pings_stop_cum > min_pings_home_cluster_label].drop('total_pings_stop_cum', axis=1)
    return home_tmp.dropna(subset=['duration_cum'])

def add_home_label(user_df, home_tmp):
    date_cluster = home_tmp.drop_duplicates(['cluster_label', 'date_trunc'])[['date_trunc', 'cluster_label']].copy()
    date_cluster = date_cluster.drop_duplicates(['date_trunc'])
    home_label = list(zip(date_cluster.cluster_label, date_cluster.date_trunc))
    idx = pd.MultiIndex.from_frame(user_df[['cluster_label', 'date_trunc']])
    user_df.loc[idx.isin(home_label), 'home_label'] = user_df.loc[idx.isin(home_label), 'cluster_label']
    return user_df


# Initialize user

(
    df
        .with_columns(
            location_type = pl.lit('O')
            , home_label = pl.lit(-1)
            , work_label = pl.lit(-1)
    )
).collect_schema()






